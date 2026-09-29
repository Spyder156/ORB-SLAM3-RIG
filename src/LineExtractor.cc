#include "LineExtractor.h"

#include <algorithm>
#include <cmath>

#include "ELSED.h"

namespace ORB_SLAM3 {

LineExtractor::LineExtractor(float minAngLenDeg, float maxAngLenDeg,
                             int gradThresh, int minLenPx)
    : mMinAngLen(minAngLenDeg * float(M_PI) / 180.f),
      mMaxAngLen(maxAngLenDeg * float(M_PI) / 180.f),
      mGateNormal(3.0f * float(M_PI) / 180.f),
      mGateDir(8.0f * float(M_PI) / 180.f),
      mGateLenRatio(0.5f),
      mGradThresh(gradThresh), mMinLenPx(minLenPx) {}

void LineExtractor::SetMask(int camIdx, const cv::Mat& mask) {
    if (camIdx < 0 || camIdx > 1) return;
    mMask[camIdx] = mask;
}

static inline bool masked(const cv::Mat& m, const cv::Point2f& p) {
    if (m.empty()) return false;
    const int x = int(p.x + 0.5f), y = int(p.y + 0.5f);
    if (x < 0 || y < 0 || x >= m.cols || y >= m.rows) return true;   // outside
    return m.at<uchar>(y, x) != 0;
}

std::vector<LineObs> LineExtractor::Extract(const cv::Mat& imGray,
                                            GeometricCamera* pCam,
                                            int camIdx) const {
    std::vector<LineObs> out;
    if (imGray.empty() || pCam == nullptr) {
        return out;
    }
    upm::ELSEDParams p;
    p.gradientThreshold = mGradThresh;
    p.minLineLen = mMinLenPx;
    upm::ELSED elsed(p);
    upm::Segments segs = elsed.detect(imGray);
    out.reserve(segs.size());

    mpCamForMerge = pCam;
    const cv::Mat& msk = (camIdx >= 0 && camIdx <= 1) ? mMask[camIdx] : cv::Mat();
    for (const auto& s : segs) {
        const cv::Point2f a(s[0], s[1]), b(s[2], s[3]);
        // Drop if EITHER endpoint is on the occluder: a segment straddling the
        // hardware edge is half real and half not, and its normal is wrong.
        if (masked(msk, a) || masked(msk, b)) continue;
        const Eigen::Vector3f b1 = pCam->unprojectEig(a);
        const Eigen::Vector3f b2 = pCam->unprojectEig(b);
        // Reject anything that did not unproject to a sane bearing. An
        // unchecked unprojection is how OKVIS2 ended up believing two
        // back-to-back cameras overlapped.
        if (!b1.allFinite() || !b2.allFinite()) continue;
        const float n1 = b1.norm(), n2 = b2.norm();
        if (n1 < 1e-6f || n2 < 1e-6f) continue;
        const Eigen::Vector3f u1 = b1 / n1, u2 = b2 / n2;

        Eigen::Vector3f n = u1.cross(u2);
        const float ln = n.norm();
        if (ln < 1e-7f) continue;                     // endpoints collinear
        const float ang = std::asin(std::min(1.0f, ln));
        // Angular, not pixel, length. A fisheye compresses the rim, so a
        // pixel-short segment there can span a large angle -- and those are the
        // wide-baseline lines that constrain rotation best.
        // The upper bound matters too: ELSED finds lines straight in the IMAGE,
        // but a straight WORLD line is curved on a fisheye, so over a long arc
        // the endpoints' great circle no longer contains the middle. Measured
        // at ~4e-3 rad of residual for a 37 deg segment.
        if (ang < mMinAngLen || ang > mMaxAngLen) continue;

        LineObs lo;
        lo.p1 = a; lo.p2 = b;
        lo.b1u = u1; lo.b2u = u2;
        lo.n = n / ln;
        Eigen::Vector3f d = u2 - u1;
        const float dn = d.norm();
        if (dn < 1e-9f) continue;
        lo.dir = d / dn;
        lo.angLen = ang;
    lo.pxPerRad = ang > 1e-6f ? float(cv::norm(a - b)) / ang : 400.f;
        lo.cam = camIdx;
        out.push_back(lo);
    }
    if (mbMergeCircles) out = MergeGreatCircles(out);
    // canonical endpoint order + appearance, AFTER merge so merged arcs get
    // their own descriptor. The polyline is sampled from the (possibly
    // curved) projected arc when a camera is available, else the pixel chord.
    for (LineObs& o : out) {
        CanonicalizeObs(o);
        std::vector<cv::Point2f> poly;
        const int K = 16;
        if (pCam) {
            const float th = std::acos(std::max(-1.f, std::min(1.f, o.b1u.dot(o.b2u))));
            for (int k = 0; k < K; k++) {
                const float t = float(k) / (K - 1);
                Eigen::Vector3f bk = th > 1e-6f
                    ? ((std::sin((1 - t) * th) * o.b1u + std::sin(t * th) * o.b2u) / std::sin(th)).normalized()
                    : o.b1u;
                poly.push_back(pCam->project(cv::Point3f(bk.x(), bk.y(), bk.z())));
            }
        } else {
            for (int k = 0; k < K; k++) {
                const float t = float(k) / (K - 1);
                poly.push_back(o.p1 * (1 - t) + o.p2 * t);
            }
        }
        ComputeBandDescriptor(imGray, poly, o);
    }
    return out;
}

void LineExtractor::CanonicalizeObs(LineObs& o) {
    // n = b1 x b2 with n_z > 0 (n_x tie-break): detector endpoint order is
    // arbitrary, and the band descriptor's left/right side only means
    // anything once the traversal direction is fixed. Swapping endpoints
    // flips n, so fixing n's hemisphere fixes the order.
    const bool flip = (std::fabs(o.n.z()) > 1e-3f) ? (o.n.z() < 0.f)
                                                   : (o.n.x() < 0.f);
    if (flip) {
        std::swap(o.p1, o.p2);
        std::swap(o.b1u, o.b2u);
        o.n = -o.n;
        o.dir = -o.dir;
    }
}

void LineExtractor::ComputeBandDescriptor(const cv::Mat& img,
                                          const std::vector<cv::Point2f>& poly,
                                          LineObs& o) {
    // Mean intensity at {-5,-2,+2,+5} px along the LOCAL pixel normal,
    // averaged along the polyline; zero-mean unit-norm. polarity = which side
    // is brighter -- the one bit that separates the two edges of a strip.
    static const float offs[4] = {-5.f, -2.f, 2.f, 5.f};
    double acc[4] = {0, 0, 0, 0};
    int cnt = 0;
    auto sample = [&](float x, float y) -> float {
        const int xi = (int)x, yi = (int)y;
        if (xi < 0 || yi < 0 || xi + 1 >= img.cols || yi + 1 >= img.rows)
            return -1.f;
        const float fx = x - xi, fy = y - yi;
        const uchar* r0 = img.ptr<uchar>(yi), *r1 = img.ptr<uchar>(yi + 1);
        return (1-fx)*(1-fy)*r0[xi] + fx*(1-fy)*r0[xi+1]
             + (1-fx)*fy*r1[xi] + fx*fy*r1[xi+1];
    };
    for (size_t k = 0; k + 1 < poly.size(); k++) {
        cv::Point2f t = poly[k + 1] - poly[std::max<size_t>(k, 1) - 1];
        const float tn = std::hypot(t.x, t.y);
        if (tn < 1e-6f) continue;
        const cv::Point2f nrm(-t.y / tn, t.x / tn);   // left of travel
        float v[4]; bool ok = true;
        for (int j = 0; j < 4; j++) {
            v[j] = sample(poly[k].x + nrm.x * offs[j], poly[k].y + nrm.y * offs[j]);
            if (v[j] < 0.f) { ok = false; break; }
        }
        if (!ok) continue;
        for (int j = 0; j < 4; j++) acc[j] += v[j];
        cnt++;
    }
    if (cnt < 4) { o.hasDesc = false; return; }
    float m = 0;
    for (int j = 0; j < 4; j++) { acc[j] /= cnt; m += (float)acc[j]; }
    m /= 4.f;
    float nn = 0;
    for (int j = 0; j < 4; j++) { o.desc[j] = (float)acc[j] - m; nn += o.desc[j] * o.desc[j]; }
    nn = std::sqrt(nn);
    const float g = float((acc[2] + acc[3]) - (acc[0] + acc[1])) * 0.5f;
    o.contrast = std::fabs(g);
    o.polarity = g >= 0.f ? 1 : -1;
    if (nn < 1e-3f) { o.hasDesc = false; return; }    // flat: no appearance
    for (int j = 0; j < 4; j++) o.desc[j] /= nn;
    o.hasDesc = true;
}

float LineExtractor::DescDist(const float a[4], const float b[4]) {
    float s = 0;
    for (int j = 0; j < 4; j++) s += (a[j] - b[j]) * (a[j] - b[j]);
    return std::sqrt(s);
}

std::vector<LineObs> LineExtractor::MergeGreatCircles(
        const std::vector<LineObs>& in) const {
    // ONE PHYSICAL EDGE, ONE OBSERVATION.
    //
    // ELSED breaks a long edge into several segments, and it breaks it
    // DIFFERENTLY every frame. All those fragments share one great circle, so
    // the matcher -- which gates on the plane normal -- cannot tell them apart:
    // frame to frame it binds fragment 1, then fragment 2, then fragment 1
    // again. The track survives but the observed extent jumps between pieces of
    // the edge, so the depth is triangulated from observations of different
    // physical pieces. That is the "three pencils in a row" failure.
    //
    // Merge every collinear fragment into a single circle observation: same
    // plane, extent = the union arc. The matching unit becomes the CIRCLE, so
    // fragmentation and endpoint clipping stop being able to move it.
    std::vector<LineObs> out;
    if (in.empty()) return out;
    const float cN = std::cos(mMergeNormalRad);
    std::vector<bool> used(in.size(), false);

    for (size_t i = 0; i < in.size(); ++i) {
        if (used[i]) continue;
        // in-plane frame of this circle, so members can be ordered along it
        const Eigen::Vector3f n0 = in[i].n;
        Eigen::Vector3f u = in[i].b1u - n0 * n0.dot(in[i].b1u);
        if (u.norm() < 1e-6f) { out.push_back(in[i]); used[i] = true; continue; }
        u.normalize();
        const Eigen::Vector3f v = n0.cross(u);
        auto ang = [&](const Eigen::Vector3f& b) {
            return std::atan2(b.dot(v), b.dot(u));
        };
        struct Piece { float lo, hi; size_t idx; };
        std::vector<Piece> pieces;
        auto add = [&](size_t k) {
            float a1 = ang(in[k].b1u), a2 = ang(in[k].b2u);
            if (a1 > a2) std::swap(a1, a2);
            if (a2 - a1 > float(M_PI)) { const float t = a1; a1 = a2; a2 = t + 2.f*float(M_PI); }
            pieces.push_back({a1, a2, k});
        };
        add(i); used[i] = true;
        for (size_t j = i + 1; j < in.size(); ++j) {
            if (used[j] || in[j].cam != in[i].cam) continue;
            if (std::fabs(n0.dot(in[j].n)) < cN) continue;   // not the same circle
            add(j); used[j] = true;
        }
        if (pieces.size() == 1) { out.push_back(in[i]); continue; }

        // union the pieces, but only across SMALL gaps: two fragments far apart
        // on the same circle may be different edges that merely happen to be
        // collinear from here, and joining them would invent extent.
        std::sort(pieces.begin(), pieces.end(),
                  [](const Piece& a, const Piece& b) { return a.lo < b.lo; });
        size_t g0 = 0;
        while (g0 < pieces.size()) {
            float lo = pieces[g0].lo, hi = pieces[g0].hi;
            float wSum = hi - lo;
            // Sign-align EVERY normal to n0, including the group's first: a
            // line has no orientation, so a fragment detected with reversed
            // endpoint order carries -n. Accumulating the first one unaligned
            // let one reversed seed cancel the others (measured 89.9 deg
            // normal error from fragments that agree to 0.2 deg).
            const Eigen::Vector3f n00 = in[pieces[g0].idx].n;
            Eigen::Vector3f nAcc = (n00.dot(n0) < 0.f ? -n00 : n00) * (hi - lo);
            size_t g1 = g0 + 1;
            for (; g1 < pieces.size(); ++g1) {
                const float gap = pieces[g1].lo - hi;
                if (gap > mMergeMaxGap * std::max(hi - lo, pieces[g1].hi - pieces[g1].lo))
                    break;                                   // too far -- new group
                hi = std::max(hi, pieces[g1].hi);
                const float w = pieces[g1].hi - pieces[g1].lo;
                // keep the accumulated normal sign-consistent with n0
                const Eigen::Vector3f nj = in[pieces[g1].idx].n;
                nAcc += (nj.dot(n0) < 0.f ? -nj : nj) * w;
                wSum += w;
            }
            LineObs m = in[pieces[g0].idx];                  // inherit cam etc.
            Eigen::Vector3f nm = nAcc / std::max(wSum, 1e-9f);
            if (nm.norm() < 1e-7f) nm = n0; else nm.normalize();
            m.n = nm;
            // ONE consistent plane: the endpoints must lie on the plane of the
            // MERGED normal, not the seed's. Re-derive the in-plane basis on
            // nm (seed basis projected onto nm's plane), so nm.b1u == 0 by
            // construction -- previously the stored normal and the stored
            // bearings described two different planes.
            Eigen::Vector3f um = u - nm * nm.dot(u);
            if (um.norm() < 1e-6f) um = u;   // ~coincident planes: keep seed basis
            um.normalize();
            const Eigen::Vector3f vm = nm.cross(um);
            m.b1u = (um * std::cos(lo) + vm * std::sin(lo)).normalized();
            m.b2u = (um * std::cos(hi) + vm * std::sin(hi)).normalized();
            m.angLen = hi - lo;
            Eigen::Vector3f dd = m.b2u - m.b1u;
            if (dd.norm() > 1e-9f) m.dir = dd.normalized();
            // pixels of the merged arc ends, for anything that draws them
            if (mpCamForMerge) {
                m.p1 = mpCamForMerge->project(cv::Point3f(m.b1u.x(), m.b1u.y(), m.b1u.z()));
                m.p2 = mpCamForMerge->project(cv::Point3f(m.b2u.x(), m.b2u.y(), m.b2u.z()));
                m.pxPerRad = m.angLen > 1e-6f
                           ? float(cv::norm(m.p1 - m.p2)) / m.angLen : 400.f;
            }
            if (m.angLen >= mMinAngLen && m.angLen <= mMaxAngLen) out.push_back(m);
            g0 = g1;
        }
    }
    return out;
}

std::vector<int> LineExtractor::Match(const std::vector<LineObs>& cur,
                                      const std::vector<LineObs>& prev) const {
    std::vector<int> assign(cur.size(), -1);
    if (prev.empty() || cur.empty()) return assign;

    const float cN = std::cos(mGateNormal), cD = std::cos(mGateDir);
    std::vector<bool> usedPrev(prev.size(), false);

    // greedy best-first on normal alignment
    struct Cand { float score; int i, j; };
    std::vector<Cand> cands;
    cands.reserve(cur.size() * 4);
    for (size_t i = 0; i < cur.size(); ++i) {
        for (size_t j = 0; j < prev.size(); ++j) {
            if (cur[i].cam != prev[j].cam) continue;      // never mix cameras
            // sign-free: a line has no orientation, so |n_i . n_j|
            const float an = std::fabs(cur[i].n.dot(prev[j].n));
            if (an < cN) continue;
            // ANGULAR OVERLAP on the shared great circle (LF-PGVIO), not a
            // length ratio. Two DIFFERENT physical edges lying on the same
            // circle -- the parallel-neighbour failure -- have similar lengths
            // and pass a ratio test; they do not overlap. The old chord gate
            // |dir_i.dir_j| is dropped: the chord is the circle's tangent at
            // the segment MIDPOINT, so it tested where the segment sat on the
            // circle rather than whether it was the same line.
            {   // d_orth in PIXELS: how far this segment's endpoints sit off
                // the other's great circle, converted with the local image
                // scale rather than assumed constant across the fisheye.
                const float s1 = std::asin(std::min(1.f, std::fabs(prev[j].n.dot(cur[i].b1u))));
                const float s2 = std::asin(std::min(1.f, std::fabs(prev[j].n.dot(cur[i].b2u))));
                if (0.5f * (s1 + s2) * cur[i].pxPerRad > mGateOrthPx) continue;
            }
            {
                const Eigen::Vector3f n = cur[i].n;
                // in-plane frame centred on this segment
                Eigen::Vector3f u = cur[i].b1u - n * n.dot(cur[i].b1u);
                if (u.norm() < 1e-6f) continue;
                u.normalize();
                const Eigen::Vector3f v = n.cross(u);
                auto ang = [&](const Eigen::Vector3f& b) {
                    return std::atan2(b.dot(v), b.dot(u));
                };
                float a1 = ang(cur[i].b1u),  a2 = ang(cur[i].b2u);
                float a3 = ang(prev[j].b1u), a4 = ang(prev[j].b2u);
                if (a1 > a2) std::swap(a1, a2);
                if (a3 > a4) std::swap(a3, a4);
                // segments are <40 deg, so a straddle of +-pi means wraparound
                const float TWO_PI = 2.f * float(M_PI);
                if (a2 - a1 > float(M_PI)) { const float t = a1; a1 = a2; a2 = t + TWO_PI; }
                if (a4 - a3 > float(M_PI)) { const float t = a3; a3 = a4; a4 = t + TWO_PI; }
                const float len1 = a2 - a1, len2 = a4 - a3;
                if (len1 < 1e-6f || len2 < 1e-6f) continue;
                const float ov = std::min(a2, a4) - std::max(a1, a3);
                if (ov <= 0.f) continue;                       // disjoint pieces
                if (ov / std::min(len1, len2) < mGateOverlap) continue;
            }
            cands.push_back({an, int(i), int(j)});
        }
    }
    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.score > b.score; });
    for (const auto& c : cands) {
        if (assign[c.i] != -1 || usedPrev[c.j]) continue;
        assign[c.i] = c.j;
        usedPrev[c.j] = true;
    }
    return assign;
}

// ---------------------------------------------------------------------------
//                                LineTracker
// ---------------------------------------------------------------------------

/// angular overlap of two arcs on (approximately) one great circle, as a
/// fraction of the SHORTER arc; <=0 when disjoint
static float ArcOverlap(const Eigen::Vector3f& n,
                        const Eigen::Vector3f& a1, const Eigen::Vector3f& a2,
                        const Eigen::Vector3f& c1, const Eigen::Vector3f& c2) {
    Eigen::Vector3f u = a1 - n * n.dot(a1);
    if (u.norm() < 1e-6f) return -1.f;
    u.normalize();
    const Eigen::Vector3f v = n.cross(u);
    auto ang = [&](const Eigen::Vector3f& b) { return std::atan2(b.dot(v), b.dot(u)); };
    float x1 = ang(a1), x2 = ang(a2), y1 = ang(c1), y2 = ang(c2);
    if (x1 > x2) std::swap(x1, x2);
    if (y1 > y2) std::swap(y1, y2);
    const float PI2 = 2.f * float(M_PI);
    if (x2 - x1 > float(M_PI)) { const float t = x1; x1 = x2; x2 = t + PI2; }
    if (y2 - y1 > float(M_PI)) { const float t = y1; y1 = y2; y2 = t + PI2; }
    const float l1 = x2 - x1, l2 = y2 - y1;
    if (l1 < 1e-6f || l2 < 1e-6f) return -1.f;
    const float ov = std::min(x2, y2) - std::max(x1, y1);
    return ov / std::min(l1, l2);
}

std::vector<int> LineTracker::Match(std::vector<LineObs>& cur,
                                    const Eigen::Matrix3f& dR_cur_prev) {
    mStats = Stats();

    // 1. PREDICT every live track one frame forward by the gyro rotation
    //    (accumulates over missed frames), and prune the long-dead.
    std::vector<LineTrack> keep;
    keep.reserve(mTracks.size());
    for (LineTrack& t : mTracks) {
        if (t.missed > mMaxMissed) { mStats.dropped++; continue; }
        t.nPred  = (dR_cur_prev * t.nPred).normalized();
        t.b1Pred = (dR_cur_prev * t.b1Pred).normalized();
        t.b2Pred = (dR_cur_prev * t.b2Pred).normalized();
        keep.push_back(t);
    }
    mTracks.swap(keep);

    std::vector<int> asg(cur.size(), -1);
    if (mTracks.empty() || cur.empty()) return asg;

    const float cN = std::cos(mGateNormalRad);
    for (size_t i = 0; i < cur.size(); i++) {
        LineObs& o = cur[i];
        // 2. LOOSE geometric shortlist against the PREDICTED tracks, and the
        //    appearance distance for each survivor. Geometry may only say
        //    "maybe" here; the decision below is appearance.
        float best = 1e9f, second = 1e9f;   // second = best COMPETING track
        int bi = -1;
        for (size_t j = 0; j < mTracks.size(); j++) {
            const LineTrack& t = mTracks[j];
            if (t.last.cam != o.cam) continue;
            mStats.nCand++;
            if (std::fabs(o.n.dot(t.nPred)) < cN) { mStats.killNormal++; continue; }
            const float s1 = std::asin(std::min(1.f, std::fabs(t.nPred.dot(o.b1u))));
            const float s2 = std::asin(std::min(1.f, std::fabs(t.nPred.dot(o.b2u))));
            if (0.5f * (s1 + s2) * o.pxPerRad > mGateOrthPx) { mStats.killOrth++; continue; }
            if (ArcOverlap(o.n, o.b1u, o.b2u, t.b1Pred, t.b2Pred) < mGateOverlap) {
                mStats.killOverlap++; continue;
            }
            // 3. APPEARANCE. Polarity first: two edges of one strip are
            //    geometric twins with opposite polarity -- when both sides are
            //    confidently contrasted and disagree, this is a different edge.
            if (o.hasDesc && t.emaInit &&
                o.contrast > mMinContrast && std::abs(t.polSum) >= 2 &&
                o.polarity * (t.polSum > 0 ? 1 : -1) < 0) {
                mStats.killPolarity++; continue;
            }
            const float d = (o.hasDesc && t.emaInit)
                          ? LineExtractor::DescDist(o.desc, t.descEma)
                          : 0.9f;   // no appearance: weak, near the abs limit
            if (d < best) { second = best; best = d; bi = int(j); }
            else if (d < second) second = d;
        }
        if (bi < 0) continue;
        // 4. AMBIGUITY. Absolute quality (a ratio can flatter two equally bad
        //    candidates) AND distinctiveness against the best competing track.
        if (best > mDescAbsMax) { mStats.killAbs++; continue; }
        if (second < 1e8f && best / std::max(second, 1e-6f) > mDescRatioMax) {
            // 5. HISTORY may RESOLVE the ambiguity -- never force a match: it
            //    only breaks the tie toward a track whose accumulated polarity
            //    agrees with the current observation.
            const LineTrack& t = mTracks[bi];
            const bool polAgree = !(o.hasDesc && t.emaInit) ||
                (o.polarity * (t.polSum >= 0 ? 1 : -1) >= 0);
            if (!(polAgree && std::abs(t.polSum) >= 3 && best < 0.7f * mDescAbsMax)) {
                mStats.killRatio++; continue;
            }
            mStats.killHistory--;   // (negative = resolved-by-history count)
        }
        // 6. ASSIGN. Per-fragment: several fragments of one broken edge may
        //    all bind to the same track -- that is fragmentation, not a clash.
        asg[i] = bi;
        mStats.matched++;
        // inherit the track's identity into this observation
        const LineObs& pl = mTracks[bi].last;
        o.mnLineId = mTracks[bi].id;
        if (pl.pML) o.pML = pl.pML;
        if (pl.hasAnchor) {
            o.hasAnchor = true;
            o.nAnchor = pl.nAnchor; o.RAnchor = pl.RAnchor;
            o.tAnchor = pl.tAnchor; o.anchorMapVersion = pl.anchorMapVersion;
            o.anchorSigma = pl.anchorSigma;
        }
        o.nSeen = pl.nSeen + 1;
    }
    return asg;
}

void LineTracker::Commit(const std::vector<LineObs>& cur,
                         const std::vector<int>& asg) {
    // store the frame's FINAL observations (Tracking mutated them since
    // Match: anchors, landmarks, triangulation) back into their tracks;
    // the longest fragment represents a multiply-fragmented track
    std::vector<float> bestLen(mTracks.size(), -1.f);
    std::vector<int> bestIdx(mTracks.size(), -1);
    for (size_t i = 0; i < cur.size(); i++) {
        const int j = asg.size() > i ? asg[i] : -1;
        if (j < 0 || j >= (int)mTracks.size()) continue;
        if (cur[i].angLen > bestLen[j]) { bestLen[j] = cur[i].angLen; bestIdx[j] = int(i); }
    }
    for (size_t j = 0; j < mTracks.size(); j++) {
        LineTrack& t = mTracks[j];
        t.age++;
        if (bestIdx[j] < 0) { t.missed++; continue; }
        const LineObs& o = cur[bestIdx[j]];
        const long id = t.id;
        t.last = o;
        t.last.mnLineId = id;
        t.missed = 0;
        t.nPred = o.n; t.b1Pred = o.b1u; t.b2Pred = o.b2u;
        if (o.hasDesc) {
            t.polSum += o.polarity;
            if (!t.emaInit) { for (int k = 0; k < 4; k++) t.descEma[k] = o.desc[k]; t.emaInit = true; }
            else for (int k = 0; k < 4; k++) t.descEma[k] = 0.85f * t.descEma[k] + 0.15f * o.desc[k];
        }
    }
    // unmatched observations found a NEW edge: give them an identity
    for (size_t i = 0; i < cur.size(); i++) {
        if (i < asg.size() && asg[i] >= 0) continue;
        LineTrack t;
        t.id = mNextId++;
        t.last = cur[i];
        t.last.mnLineId = t.id;
        t.nPred = cur[i].n; t.b1Pred = cur[i].b1u; t.b2Pred = cur[i].b2u;
        t.age = 1; t.missed = 0;
        if (cur[i].hasDesc) {
            t.polSum = cur[i].polarity;
            for (int k = 0; k < 4; k++) t.descEma[k] = cur[i].desc[k];
            t.emaInit = true;
        }
        mTracks.push_back(t);
        mStats.fresh++;
    }
}

}  // namespace ORB_SLAM3

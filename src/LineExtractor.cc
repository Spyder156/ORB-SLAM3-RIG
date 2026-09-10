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
    return out;
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
            Eigen::Vector3f nAcc = in[pieces[g0].idx].n * (hi - lo);
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
            m.b1u = (u * std::cos(lo) + v * std::sin(lo)).normalized();
            m.b2u = (u * std::cos(hi) + v * std::sin(hi)).normalized();
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

}  // namespace ORB_SLAM3

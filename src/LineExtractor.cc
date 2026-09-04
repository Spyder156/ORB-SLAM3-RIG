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
        lo.cam = camIdx;
        out.push_back(lo);
    }
    return out;
}

void LineExtractor::ComputeLBD(const cv::Mat& img, std::vector<LineObs>& obs) const {
    // gradients once per frame
    cv::Mat gx, gy;
    cv::Sobel(img, gx, CV_32F, 1, 0, 3);
    cv::Sobel(img, gy, CV_32F, 0, 1, 3);
    const int NB = 9, BW = 7;               // 9 bands x 7 px across the line
    const int ROWS = NB * BW;
    // Gaussian across the band, as in the paper: strips near the line dominate
    std::vector<float> wg(ROWS);
    for (int r = 0; r < ROWS; r++) {
        const float off = r - (ROWS - 1) / 2.f;
        const float sg = 0.5f * ROWS / 2.f;
        wg[r] = std::exp(-off * off / (2.f * sg * sg));
    }
    for (auto& lo : obs) {
        const float dx = lo.p2.x - lo.p1.x, dy = lo.p2.y - lo.p1.y;
        const float L = std::sqrt(dx * dx + dy * dy);
        if (L < 2.f) { lo.hasLbd = false; continue; }
        const float ux = dx / L, uy = dy / L;      // along
        const float vx = -uy, vy = ux;             // across
        const int NA = std::max(5, std::min(40, (int)L));
        float d[72] = {0};
        for (int b = 0; b < NB; b++) {
            float m[4] = {0,0,0,0}, m2[4] = {0,0,0,0};
            int cnt = 0;
            for (int r = b * BW; r < (b + 1) * BW; r++) {
                const float off = r - (ROWS - 1) / 2.f;
                for (int a = 0; a < NA; a++) {
                    const float t = NA == 1 ? 0.f : (float)a / (NA - 1);
                    const int x = (int)std::lround(lo.p1.x + t * dx + off * vx);
                    const int y = (int)std::lround(lo.p1.y + t * dy + off * vy);
                    if (x < 0 || y < 0 || x >= img.cols || y >= img.rows) continue;
                    const float GX = gx.at<float>(y, x), GY = gy.at<float>(y, x);
                    const float dL = (GX * ux + GY * uy) * wg[r];
                    const float dO = (GX * vx + GY * vy) * wg[r];
                    const float f[4] = {std::max(dL, 0.f), std::max(-dL, 0.f),
                                        std::max(dO, 0.f), std::max(-dO, 0.f)};
                    for (int c = 0; c < 4; c++) { m[c] += f[c]; m2[c] += f[c] * f[c]; }
                    cnt++;
                }
            }
            if (cnt < 4) continue;
            for (int c = 0; c < 4; c++) {
                const float mu = m[c] / cnt;
                d[b * 8 + c] = mu;
                d[b * 8 + 4 + c] = std::sqrt(std::max(0.f, m2[c] / cnt - mu * mu));
            }
        }
        float n = 0; for (int i = 0; i < 72; i++) n += d[i] * d[i];
        n = std::sqrt(n);
        if (n < 1e-9f) { lo.hasLbd = false; continue; }
        for (int i = 0; i < 72; i++) d[i] = std::min(d[i] / n, 0.2f);  // SIFT clamp
        n = 0; for (int i = 0; i < 72; i++) n += d[i] * d[i];
        n = std::sqrt(n);
        for (int i = 0; i < 72; i++) lo.lbd[i] = d[i] / n;
        lo.hasLbd = true;
    }
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
            const float ad = std::fabs(cur[i].dir.dot(prev[j].dir));
            if (ad < cD) continue;
            const float lo = std::min(cur[i].angLen, prev[j].angLen);
            const float hi = std::max(cur[i].angLen, prev[j].angLen);
            if (hi < 1e-9f || lo / hi < mGateLenRatio) continue;
            // LBD appearance gate + ranking. Geometry alone cannot tell one
            // door-frame edge from its parallel neighbour; the descriptor can.
            // Offline on this dataset: bad matches -47%, good kept 92%.
            if (cur[i].hasLbd && prev[j].hasLbd) {
                float dd = 0;
                for (int q = 0; q < 72; q++) {
                    const float e = cur[i].lbd[q] - prev[j].lbd[q];
                    dd += e * e;
                }
                dd = std::sqrt(dd);
                if (dd > 0.55f) continue;
                cands.push_back({2.f - dd, int(i), int(j)});   // best appearance first
            } else
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

#include "MapLine.h"
#include "Map.h"
#include "KeyFrame.h"
#include "MapPoint.h"

#include <cmath>

// KeyFrame/Map are only held as forward-declared POINTERS here, so their full
// headers are not needed -- and including them drags in Pangolin/DBoW2, which
// makes this unit-testable only inside the full link.

namespace ORB_SLAM3 {

long unsigned int MapLine::nNextId = 0;
// Minimum sin^2 of the angle between an observed bearing and the line before
// their intersection is trusted. 0.25 = 30 deg (PLVS), 0.0194 = 8 deg (old).
float MapLine::kMinSinSqViewAngle = 0.25f;
bool MapLine::kVoteInPose = true;
// Extent-gate audit: which test refuses to record where a line was seen.
// A refused observation leaves the landmark with NO extent, so it is dropped
// from the map dump entirely -- a systematic filter on WHICH GEOMETRY the map
// is allowed to contain, not a neutral safety check.
std::atomic<long> MapLine::nRejParallel{0}, MapLine::nRejDepth{0},
                  MapLine::nRejRatio{0}, MapLine::nRejLong{0},
                  MapLine::nAccepted{0};

MapLine::MapLine(const Eigen::Vector3f& d, const Eigen::Vector3f& m,
                 KeyFrame* pRefKF, Map* pMap)
    : mpRefKF(pRefKF), mpMap(pMap) {
    SetPlucker(d, m);
    mnId = nNextId++;
}

void MapLine::SetEndpoints(const Eigen::Vector3f& e1, const Eigen::Vector3f& e2) {
    std::unique_lock<std::mutex> lk(mMutexPos);
    mEnd1 = e1; mEnd2 = e2; mbHasExtent = true;
    // (d,m) is a CACHE derived from the endpoints, so every existing consumer
    // (matching gates, NormalInCamera, AngularError) keeps working unchanged.
    Eigen::Vector3f d = e2 - e1;
    const float n = d.norm();
    if (n > 1e-9f) {
        mDir = d / n;
        mMom = e1.cross(mDir);          // moment about the world origin
    }
}

void MapLine::GetEndpoints(Eigen::Vector3f& e1, Eigen::Vector3f& e2) {
    std::unique_lock<std::mutex> lk(mMutexPos);
    e1 = mEnd1; e2 = mEnd2;
}

void MapLine::SetPlucker(const Eigen::Vector3f& d, const Eigen::Vector3f& m) {
    std::unique_lock<std::mutex> lk(mMutexPos);
    const float dn = d.norm();
    mDir = (dn > 1e-9f) ? (d / dn).eval() : Eigen::Vector3f(0, 0, 1);
    // Plucker constraint: the moment must be orthogonal to the direction.
    // Numerical drift breaks that, and a non-orthogonal pair is not a line at
    // all, so re-project every time rather than trusting the caller.
    mMom = m - mDir * (mDir.dot(m));
    // THE EXTENT BELONGS TO THE LINE. Whenever (d,m) moves -- every Local BA
    // refinement -- endpoints recorded against the OLD line are no longer on
    // this one, and everything that draws or reasons about extent then places
    // the segment where the line is not (measured: 24% of landmarks adrift,
    // worst 35 m). Slide them onto the new line along its direction: the
    // observed interval is preserved, the geometry stays consistent.
    if (mbHasExtent) {
        const Eigen::Vector3f p0 = mDir.cross(mMom);   // closest point to origin
        mEnd1 = p0 + mDir * (mEnd1 - p0).dot(mDir);
        mEnd2 = p0 + mDir * (mEnd2 - p0).dot(mDir);
    }
}

Eigen::Vector3f MapLine::GetDirection() {
    std::unique_lock<std::mutex> lk(mMutexPos);
    return mDir;
}

Eigen::Vector3f MapLine::GetMoment() {
    std::unique_lock<std::mutex> lk(mMutexPos);
    return mMom;
}

Eigen::Vector3f MapLine::NormalInCamera(const Eigen::Matrix3f& Rcw,
                                        const Eigen::Vector3f& tcw) {
    Eigen::Vector3f d, m;
    {
        std::unique_lock<std::mutex> lk(mMutexPos);
        d = mDir; m = mMom;
    }
    // Plucker transform world -> camera: m_c = R m + t x (R d)
    const Eigen::Vector3f n = Rcw * m + tcw.cross(Rcw * d);
    const float nn = n.norm();
    return (nn > 1e-9f) ? (n / nn).eval() : Eigen::Vector3f::Zero();
}

float MapLine::AngularError(const Eigen::Matrix3f& Rcw, const Eigen::Vector3f& tcw,
                            const Eigen::Vector3f& bearing) {
    const Eigen::Vector3f n = NormalInCamera(Rcw, tcw);
    if (n.squaredNorm() < 1e-12f) return float(M_PI);
    const float bn = bearing.norm();
    if (bn < 1e-9f) return float(M_PI);
    // |n . b| is the sine of the angle between the bearing and the plane
    return std::asin(std::min(1.0f, std::fabs(n.dot(bearing / bn))));
}

void MapLine::ApplyScaledRotation(const Eigen::Matrix3f& R, float s,
                                  const Eigen::Vector3f& t) {
    std::unique_lock<std::mutex> lk(mMutexPos);
    const Eigen::Vector3f d2 = R * mDir;
    mMom = s * (R * mMom) + t.cross(d2);
    mDir = d2;
    if (mbHasExtent) {
        mEnd1 = s * (R * mEnd1) + t;
        mEnd2 = s * (R * mEnd2) + t;
    }
    mbHasFirst = false;   // stored first-obs pose is in the old frame
}

void MapLine::UnionExtent(const Eigen::Vector3f& e1, const Eigen::Vector3f& e2) {
    std::unique_lock<std::mutex> lk(mMutexPos);
    if (!mbHasExtent) { mEnd1 = e1; mEnd2 = e2; mbHasExtent = true; return; }
    // parameter along the line through mEnd1 with direction mDir
    const Eigen::Vector3f p0 = mEnd1;
    float tmin = 0.f, tmax = (mEnd2 - p0).dot(mDir);
    if (tmax < tmin) std::swap(tmin, tmax);
    for (const Eigen::Vector3f* e : {&e1, &e2}) {
        const float t = (*e - p0).dot(mDir);
        tmin = std::min(tmin, t); tmax = std::max(tmax, t);
    }
    mEnd1 = p0 + tmin * mDir;
    mEnd2 = p0 + tmax * mDir;
}

bool MapLine::SetExtentFromBearings(const Eigen::Matrix3f& Rcw,
                                    const Eigen::Vector3f& tcw,
                                    const Eigen::Vector3f& b1,
                                    const Eigen::Vector3f& b2,
                                    float maxDepth) {
    Eigen::Vector3f d_w, m_w;
    { std::unique_lock<std::mutex> lk(mMutexPos); d_w = mDir; m_w = mMom; }
    const Eigen::Vector3f d_c = Rcw * d_w;
    const Eigen::Vector3f m_c = Rcw * m_w + tcw.cross(d_c);

    float s1v = 0.f, s2v = 0.f;
    int why = 0;   // which gate refused: 1 near-parallel, 2 depth range
    auto onLine = [&](const Eigen::Vector3f& b, Eigen::Vector3f& x_w, float& sv) {
        const Eigen::Vector3f cr = b.cross(d_c);
        const float den = cr.squaredNorm();
        // sin(bearing, line) >= sin(8 deg): nearly-parallel geometry amplifies
        // moment noise by 1/sin^2 and produced tens-of-metre radial spikes
        // sin^2 of the angle between the bearing and the line. At 8 deg the
        // intersection is amplified 7x; a line pointing near the view ray then
        // gets an absurd extent that radiates away from the camera -- which is
        // what the map's long segments are. PLVS refuses anything within 30 deg
        // (kCosViewZAngleMax = cos 30). sin^2(30) = 0.25.
        if (!std::isfinite(den) || den < kMinSinSqViewAngle) { why = 1; return false; }
        const float s = m_c.dot(cr) / den;
        if (!std::isfinite(s) || s <= 0.05f || s > maxDepth) { why = 2; return false; }
        sv = s;
        x_w = Rcw.transpose() * (s * b - tcw);
        return x_w.allFinite();
    };
    Eigen::Vector3f e1, e2;
    if (!onLine(b1, e1, s1v) || !onLine(b2, e2, s2v)) {
        (why == 1 ? nRejParallel : nRejDepth)++;
        return false;
    }
    // the two endpoint depths of one observed segment cannot differ wildly,
    // and an indoor segment is not tens of metres long
    if (std::max(s1v, s2v) > 5.f * std::min(s1v, s2v)) { nRejRatio++; return false; }
    if ((e2 - e1).norm() > 20.f) { nRejLong++; return false; }
    nAccepted++;
    std::unique_lock<std::mutex> lk(mMutexPos);
    if (!mbHasExtent) {
        mEnd1 = e1; mEnd2 = e2; mbHasExtent = true; mnExtentObs = 1;
    } else {
        // AVERAGE of the observed intervals, not the union. Union only ever
        // grows, so one spuriously long detection permanently stretches the
        // segment past the physical edge; the running mean converges to the
        // typically-observed extent instead.
        // Parameterise both intervals along mDir from p0, low end to high end.
        const Eigen::Vector3f p0 = mEnd1;
        float lo0 = 0.f, hi0 = (mEnd2 - p0).dot(mDir);
        if (hi0 < lo0) std::swap(lo0, hi0);
        float lo1 = (e1 - p0).dot(mDir), hi1 = (e2 - p0).dot(mDir);
        if (hi1 < lo1) std::swap(lo1, hi1);
        const float n = float(mnExtentObs);
        const float lo = (lo0 * n + lo1) / (n + 1.f);
        const float hi = (hi0 * n + hi1) / (n + 1.f);
        mEnd1 = p0 + lo * mDir;
        mEnd2 = p0 + hi * mDir;
        mnExtentObs++;
    }
    return true;
}

bool MapLine::Triangulate(const Eigen::Vector3f& n1_c, const Eigen::Matrix3f& Rcw1,
                          const Eigen::Vector3f& tcw1,
                          const Eigen::Vector3f& n2_c, const Eigen::Matrix3f& Rcw2,
                          const Eigen::Vector3f& tcw2,
                          float minAngleDeg,
                          Eigen::Vector3f& d_w, Eigen::Vector3f& m_w) {
    // interpretation planes, expressed in the world frame
    const Eigen::Vector3f n1_w = Rcw1.transpose() * n1_c;
    const Eigen::Vector3f n2_w = Rcw2.transpose() * n2_c;
    const Eigen::Vector3f C1 = -Rcw1.transpose() * tcw1;
    const Eigen::Vector3f C2 = -Rcw2.transpose() * tcw2;

    Eigen::Vector3f d = n1_w.cross(n2_w);
    const float dn = d.norm();
    // Nearly parallel planes mean the intersection is unconditioned. Refusing
    // here is the whole point: a line seen twice from almost the same viewpoint
    // carries no depth, and pretending otherwise creates a landmark at a
    // arbitrary distance that then poisons the map.
    const float sinMin = std::sin(minAngleDeg * float(M_PI) / 180.f);
    if (!std::isfinite(dn) || dn < sinMin) return false;
    d /= dn;

    // A point on the line: intersect plane1, plane2 and the plane through the
    // two camera centres, which pins the free parameter along d.
    Eigen::Matrix3f A;
    A.row(0) = n1_w.transpose();
    A.row(1) = n2_w.transpose();
    A.row(2) = d.transpose();
    Eigen::Vector3f b(n1_w.dot(C1), n2_w.dot(C2), 0.f);
    const Eigen::FullPivLU<Eigen::Matrix3f> lu(A);
    if (!lu.isInvertible()) return false;
    const Eigen::Vector3f P = lu.solve(b);
    if (!P.allFinite()) return false;

    d_w = d;
    m_w = P.cross(d);          // moment of the line through P with direction d
    return true;
}

void MapLine::AddObservation(KeyFrame* pKF, int idx) {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    mObservations[pKF] = idx;
}

void MapLine::EraseObservation(KeyFrame* pKF) {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    mObservations.erase(pKF);
    if (mObservations.size() < 2) mbBad = true;
}

std::map<KeyFrame*, int> MapLine::GetObservations() {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    return mObservations;
}

int MapLine::Observations() {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    return int(mObservations.size());
}

void MapLine::AddSupportPoint(MapPoint* pMP) {
    if (!pMP || pMP->isBad()) return;
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    for (MapPoint* q : mvpSupport) if (q == pMP) return;
    if (mvpSupport.size() >= 80) return;          // enough to fit a line
    mvpSupport.push_back(pMP);
}

int MapLine::SupportCount() {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    int n = 0;
    for (MapPoint* q : mvpSupport) if (q && !q->isBad()) n++;
    return n;
}

bool MapLine::RefitFromPoints(float inlierTol, int minInliers, float minSpan) {
    // A PAIR IS NEVER TRUSTED ALONE. Any two well-separated points define a
    // line with themselves as its only inliers, so minInliers=2 re-admits the
    // doorway failure (near point + far point on one viewing circle -> a long
    // chord through empty space). Three points must AGREE before the fit is
    // geometry, and the consensus must cover most of the support set --
    // otherwise this "line" is a mixture of different physical edges (the
    // three-pencils case) and updating from it would be fiction.
    minInliers = std::max(minInliers, 3);
    // copy live support positions (prune dead ones while at it)
    std::vector<Eigen::Vector3f> P;
    {
        std::unique_lock<std::mutex> lk(mMutexFeatures);
        std::vector<MapPoint*> keep;
        for (MapPoint* q : mvpSupport) {
            if (!q || q->isBad()) continue;
            keep.push_back(q);
            P.push_back(q->GetWorldPos());
        }
        mvpSupport.swap(keep);
    }
    if ((int)P.size() < minInliers) { mbSupportFitOk = false; return false; }

    // best collinear pair (points share a great circle without sharing an
    // edge -- e.g. near wall + through a doorway -- so a plain PCA over all
    // of them fits the outlier geometry; find the consensus line first)
    int bi = -1, bj = -1, bestIn = 0;
    for (size_t i = 0; i < P.size(); i++)
        for (size_t j = i + 1; j < P.size(); j++) {
            Eigen::Vector3f d = P[j] - P[i];
            const float dn = d.norm();
            if (dn < minSpan) continue;
            d /= dn;
            int nin = 0;
            for (const auto& X : P)
                if ((X - P[i]).cross(d).norm() < inlierTol) nin++;
            if (nin > bestIn) { bestIn = nin; bi = int(i); bj = int(j); }
        }
    if (bi < 0 || bestIn < minInliers) { mnFitFail++; mbSupportFitOk = false; return false; }
    if (float(bestIn) < 0.6f * float(P.size())) { mnFitFail++; mbSupportFitOk = false; return false; }
    mnFitFail = 0;

    Eigen::Vector3f d0 = (P[bj] - P[bi]).normalized();
    // PCA refine over the inliers
    Eigen::Vector3f c = Eigen::Vector3f::Zero(); int nin = 0;
    for (const auto& X : P)
        if ((X - P[bi]).cross(d0).norm() < inlierTol) { c += X; nin++; }
    c /= float(nin);
    Eigen::Matrix3f C = Eigen::Matrix3f::Zero();
    for (const auto& X : P)
        if ((X - P[bi]).cross(d0).norm() < inlierTol) {
            const Eigen::Vector3f q = X - c;
            C += q * q.transpose();
        }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(C);
    Eigen::Vector3f d = es.eigenvectors().col(2);      // largest eigenvalue
    if (d.dot(d0) < 0.f) d = -d;
    // endpoints = inlier span along the fitted axis
    float tmin = 1e9f, tmax = -1e9f;
    for (const auto& X : P) {
        if ((X - c).cross(d).norm() >= inlierTol) continue;
        const float t = (X - c).dot(d);
        tmin = std::min(tmin, t); tmax = std::max(tmax, t);
    }
    if (!(tmax - tmin >= minSpan)) { mbSupportFitOk = false; return false; }
    SetEndpoints(c + d * tmin, c + d * tmax);
    mbSupportFitOk = true;
    return true;
}

void MapLine::SetBadFlag() {
    // Mirror MapPoint::SetBadFlag: drop the landmark from every keyframe that
    // holds it AND from the map, not just flag it. A flagged-but-reachable
    // landmark keeps being drawn, keeps entering BA, and keeps voting.
    std::map<KeyFrame*, int> obs;
    {
        std::unique_lock<std::mutex> lk(mMutexFeatures);
        std::unique_lock<std::mutex> lk2(mMutexPos);
        mbBad = true;
        obs = mObservations;
        mObservations.clear();
    }
    for (std::map<KeyFrame*, int>::iterator it = obs.begin(); it != obs.end(); it++)
        if (it->first) it->first->EraseMapLineMatch(this);
    if (mpMap) mpMap->EraseMapLine(this);
}

bool MapLine::isBad() {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    return mbBad;
}

}  // namespace ORB_SLAM3

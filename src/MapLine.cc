#include "MapLine.h"

#include <cmath>

// KeyFrame/Map are only held as forward-declared POINTERS here, so their full
// headers are not needed -- and including them drags in Pangolin/DBoW2, which
// makes this unit-testable only inside the full link.

namespace ORB_SLAM3 {

long unsigned int MapLine::nNextId = 0;

MapLine::MapLine(const Eigen::Vector3f& d, const Eigen::Vector3f& m,
                 KeyFrame* pRefKF, Map* pMap)
    : mpRefKF(pRefKF), mpMap(pMap) {
    SetPlucker(d, m);
    mnId = nNextId++;
}

void MapLine::SetPlucker(const Eigen::Vector3f& d, const Eigen::Vector3f& m) {
    std::unique_lock<std::mutex> lk(mMutexPos);
    const float dn = d.norm();
    mDir = (dn > 1e-9f) ? (d / dn).eval() : Eigen::Vector3f(0, 0, 1);
    // Plucker constraint: the moment must be orthogonal to the direction.
    // Numerical drift breaks that, and a non-orthogonal pair is not a line at
    // all, so re-project every time rather than trusting the caller.
    mMom = m - mDir * (mDir.dot(m));
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
    auto onLine = [&](const Eigen::Vector3f& b, Eigen::Vector3f& x_w, float& sv) {
        const Eigen::Vector3f cr = b.cross(d_c);
        const float den = cr.squaredNorm();
        // sin(bearing, line) >= sin(8 deg): nearly-parallel geometry amplifies
        // moment noise by 1/sin^2 and produced tens-of-metre radial spikes
        if (!std::isfinite(den) || den < 0.0194f) return false;
        const float s = m_c.dot(cr) / den;
        if (!std::isfinite(s) || s <= 0.05f || s > maxDepth) return false;
        sv = s;
        x_w = Rcw.transpose() * (s * b - tcw);
        return x_w.allFinite();
    };
    Eigen::Vector3f e1, e2;
    if (!onLine(b1, e1, s1v) || !onLine(b2, e2, s2v)) return false;
    // the two endpoint depths of one observed segment cannot differ wildly,
    // and an indoor segment is not tens of metres long
    if (std::max(s1v, s2v) > 5.f * std::min(s1v, s2v)) return false;
    if ((e2 - e1).norm() > 20.f) return false;
    std::unique_lock<std::mutex> lk(mMutexPos);
    if (!mbHasExtent) {
        mEnd1 = e1; mEnd2 = e2; mbHasExtent = true;
    } else {
        // UNION with the stored extent: every sighting extends the landmark
        // along its direction instead of replacing it, so many short
        // observations of one physical edge accumulate into one long segment.
        const Eigen::Vector3f p0 = mEnd1;
        float tmin = 0.f, tmax = (mEnd2 - p0).dot(mDir);
        if (tmax < tmin) std::swap(tmin, tmax);
        for (const Eigen::Vector3f* e : {&e1, &e2}) {
            const float t = (*e - p0).dot(mDir);
            tmin = std::min(tmin, t); tmax = std::max(tmax, t);
        }
        if (tmax - tmin <= 25.f) {          // sanity: no runaway growth
            mEnd1 = p0 + tmin * mDir;
            mEnd2 = p0 + tmax * mDir;
        }
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

void MapLine::SetBadFlag() {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    mbBad = true;
    mObservations.clear();
}

bool MapLine::isBad() {
    std::unique_lock<std::mutex> lk(mMutexFeatures);
    return mbBad;
}

}  // namespace ORB_SLAM3

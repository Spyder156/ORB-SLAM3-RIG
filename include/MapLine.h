/**
 * MapLine.h -- a 3D line landmark.
 *
 * PARAMETERISATION. A 3D line has 4 DoF. Storing two endpoints (6 numbers)
 * over-parameterises it and lets the endpoints slide along the line during
 * optimisation without changing the residual -- a gauge freedom that makes the
 * normal equations singular. We store the ORTHONORMAL representation
 * (Bartoli & Sturm, CVIU 2005): a rotation U in SO(3) and an angle w, from
 * which the Plucker coordinates are
 *
 *     direction d = U * e3 ,   moment m = cos(w)/sin(w) * (U * e1)
 *
 * That is exactly 4 DoF, minimal and singularity-free.
 *
 * RESIDUAL. For a camera with pose (Rcw, tcw), the line's moment in the camera
 * frame gives the great-circle normal
 *
 *     n_c = normalize( Rcw * m + tcw x (Rcw * d) )
 *
 * and an observed bearing b on the line satisfies  n_c . b = 0. We use the
 * angular distance of each observed ENDPOINT bearing to that plane. It is
 * invariant to sliding along the line, so we never pretend to know where along
 * the edge we are -- the aperture problem, respected rather than fought.
 */
#ifndef MAPLINE_H
#define MAPLINE_H

#include <map>
#include <mutex>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace ORB_SLAM3 {

class KeyFrame;
class Map;

class MapLine {
public:
    MapLine(const Eigen::Vector3f& d, const Eigen::Vector3f& m,
            KeyFrame* pRefKF, Map* pMap);

    /// Plucker (direction, moment), always kept normalised and orthogonal.
    Eigen::Vector3f GetDirection();
    Eigen::Vector3f GetMoment();
    void SetPlucker(const Eigen::Vector3f& d, const Eigen::Vector3f& m);

    /// Great-circle normal this line should produce in a camera at (Rcw,tcw).
    Eigen::Vector3f NormalInCamera(const Eigen::Matrix3f& Rcw,
                                   const Eigen::Vector3f& tcw);

    /// Angular distance [rad] of a bearing from the line's plane. This is the
    /// residual the optimiser minimises.
    float AngularError(const Eigen::Matrix3f& Rcw, const Eigen::Vector3f& tcw,
                       const Eigen::Vector3f& bearing);

    /// Triangulate from two observations of the SAME line: the 3D line is the
    /// intersection of the two interpretation planes, so its direction is
    /// n1 x n2 in world coordinates. Returns false when the planes are too
    /// nearly parallel for the intersection to be conditioned -- which is the
    /// normal case for a line seen twice from almost the same viewpoint.
    static bool Triangulate(const Eigen::Vector3f& n1_c, const Eigen::Matrix3f& Rcw1,
                            const Eigen::Vector3f& tcw1,
                            const Eigen::Vector3f& n2_c, const Eigen::Matrix3f& Rcw2,
                            const Eigen::Vector3f& tcw2,
                            float minAngleDeg,
                            Eigen::Vector3f& d_w, Eigen::Vector3f& m_w);

    KeyFrame* GetReferenceKeyFrame() const { return mpRefKF; }
    /// Re-express this line under the map transform x' = s*R*x + t (the same
    /// one ApplyScaledRotation applies to every MapPoint). For a point p on
    /// the line:  p' = s*R*p + t,  d' = R*d,
    ///   m' = p' x d' = s*R*(p x d) + t x (R*d) = s*R*m + t x d'.
    void ApplyScaledRotation(const Eigen::Matrix3f& R, float s,
                             const Eigen::Vector3f& t);

    void AddObservation(KeyFrame* pKF, int idx);
    void EraseObservation(KeyFrame* pKF);
    std::map<KeyFrame*, int> GetObservations();
    int Observations();

    void SetBadFlag();
    bool isBad();

    long unsigned int mnId;
    static long unsigned int nNextId;
    long unsigned int mnBALocalForKF = 0;
    int mnVisible = 1, mnFound = 1;
    /// Angular extent of the observation that created this line [rad]. Only
    /// used to draw a segment of sensible length -- the line itself is infinite.
    float mAngLen = 0.05f;
    /// Parallax of the two interpretation planes at creation [rad] -- the
    /// conditioning of this landmark's triangulation.
    float mCreateParallax = 0.f;
    /// Validated reobservations survived (angular gate + cheirality). A
    /// 2-view line is a hypothesis; it becomes a constraint only after
    /// independent confirmation -- the analogue of MapPointCulling's
    /// 3-observation rule for points.
    int mnValidated = 0;

    /// First observation (plane normal + lens pose), kept so the line can be
    /// RE-TRIANGULATED against later, wider-baseline observations -- the
    /// line's version of what BA does for points. Invalidated when the world
    /// frame is re-expressed (the stored pose would be in the old frame).
    Eigen::Vector3f mFirstN = Eigen::Vector3f::Zero();
    Eigen::Matrix3f mFirstR = Eigen::Matrix3f::Identity();
    Eigen::Vector3f mFirstT = Eigen::Vector3f::Zero();
    bool mbHasFirst = false;

    /// Extend the stored extent with a newly observed piece of the line:
    /// parameterise all endpoints along the direction and keep the outermost.
    /// Fifty sightings of one door frame become ONE long segment.
    void UnionExtent(const Eigen::Vector3f& e1, const Eigen::Vector3f& e2);

    /// The piece of this (infinite) line that was actually OBSERVED, in world
    /// coordinates. A Plucker line has no endpoints, so anything that draws or
    /// reasons about extent has to be told where the camera saw it -- guessing
    /// (e.g. the point of closest approach) puts the segment somewhere the
    /// camera never looked.
    Eigen::Vector3f mEnd1 = Eigen::Vector3f::Zero();
    Eigen::Vector3f mEnd2 = Eigen::Vector3f::Zero();
    bool mbHasExtent = false;

    /// Intersect the two endpoint bearings with this line to recover the
    /// observed segment. For bearing b and line (d_c, m_c) in camera
    /// coordinates the point on both is  s*b  with
    ///     s = (m_c . (b x d_c)) / |b x d_c|^2
    /// Returns false when the bearing is near-parallel to the line (the
    /// intersection is unconditioned) or the depth is not physical.
    bool SetExtentFromBearings(const Eigen::Matrix3f& Rcw, const Eigen::Vector3f& tcw,
                               const Eigen::Vector3f& b1, const Eigen::Vector3f& b2,
                               float maxDepth = 100.f);

protected:
    Eigen::Vector3f mDir;      ///< unit direction
    Eigen::Vector3f mMom;      ///< moment, orthogonal to mDir
    std::map<KeyFrame*, int> mObservations;
    KeyFrame* mpRefKF;
    Map* mpMap;
    bool mbBad = false;
    std::mutex mMutexPos, mMutexFeatures;
};

}  // namespace ORB_SLAM3
#endif

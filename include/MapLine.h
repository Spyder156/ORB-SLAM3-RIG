/**
 * MapLine.h -- a 3D line landmark.
 *
 * PARAMETERISATION. TWO 3D ENDPOINTS in the world frame (6 numbers); the
 * Plucker pair (d,m) is a cache derived from them. This is what every
 * published point+line SLAM uses -- PLVS VertexSBALine<6>, ORB-LINE-SLAM
 * VertexSBALineXYZ, Structure-SLAM / RGBD-PL-SLAM VertexSBAPointXYZ per
 * endpoint.
 *
 * We previously used the minimal 4-DoF orthonormal form (Bartoli & Sturm).
 * It is elegant but its residual has to be built from the line's moment ABOUT
 * THE CAMERA CENTRE, n = R m + t x (R d), which must be normalised -- and |n|
 * IS the camera-to-line distance. Normalising deletes it, so a line threaded
 * through the camera centres scores EXACTLY ZERO on every observation and the
 * optimiser prefers it to the truth. Endpoints have no such escape: an
 * endpoint at a camera centre has no bearing and is killed by cheirality.
 *
 * RESIDUAL (see EdgeLine / EdgeLineOnlyPose). Project each endpoint into the
 * observing lens and take its angular distance from the OBSERVED great circle,
 * scaled to pixels. The measurement is the observed plane normal (a constant
 * from the detector); the prediction is the endpoint bearing. Sliding an
 * endpoint ALONG the line leaves it on the same great circle, so the residual
 * is blind to it -- the aperture problem, respected. That blindness is gauge:
 * BA would happily slide both endpoints together until the segment collapses
 * to a point, so the extent is RE-DERIVED from the observations after every
 * BA writeback rather than left as a free parameter.
 */
#ifndef MAPLINE_H
#define MAPLINE_H

#include <atomic>
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
    /// DERIVED from the endpoints -- the endpoints are the state now (see the
    /// note at the top of this header).
    Eigen::Vector3f GetDirection();
    Eigen::Vector3f GetMoment();
    void SetPlucker(const Eigen::Vector3f& d, const Eigen::Vector3f& m);

    /// THE STATE. Two 3D endpoints in the WORLD frame; (d,m) is recomputed
    /// from them. This is what every published point+line SLAM optimises
    /// (PLVS VertexSBALine<6>, ORB-LINE-SLAM VertexSBALineXYZ,
    /// Structure-SLAM / RGBD-PL-SLAM VertexSBAPointXYZ per endpoint).
    void SetEndpoints(const Eigen::Vector3f& e1, const Eigen::Vector3f& e2);
    void GetEndpoints(Eigen::Vector3f& e1, Eigen::Vector3f& e2);

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
    /// Extent-gate audit counters (see MapLine.cc).
    static std::atomic<long> nRejParallel, nRejDepth, nRejRatio, nRejLong, nAccepted;
    long unsigned int mnBALocalForKF = 0;
    int mnVisible = 1, mnFound = 1;
    /// Fraction of the frames in which this landmark was PREDICTED to be
    /// visible where a segment actually bound to it. The line analogue of
    /// MapPoint::GetFoundRatio(); MapLineCulling kills anything below 0.25,
    /// exactly as PLVS does.
    float GetFoundRatio() const { return float(mnFound) / std::max(1, mnVisible); }
    /// Keyframe this landmark was first attached to (-1 until then). Used by
    /// MapLineCulling to give a new landmark a couple of keyframes to prove
    /// itself before demanding a minimum observation count.
    long int mnFirstKFid = -1;
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

    /// Rig lens that created this line (0 front, 1 rear). Re-acquisition
    /// only proposes a landmark to segments of ITS lens -- matching never
    /// crosses lenses anywhere in the pipeline.
    int mnCam = 0;
    // Re-acquisition bookkeeping, TRACKING THREAD ONLY (no lock):
    long mnLastFrameSeen = -1;     ///< frame id this landmark last bound a segment
    bool mbInReacqPool = false;    ///< currently in Tracking's recent-lines pool
    long mnValidatedFrameId = -1;  ///< guard: re-acq validates once per frame

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
    /// Sightings folded into the extent average (viz-only, like the extent).
    int mnExtentObs = 0;

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

/**
 * LineExtractor.h -- spherical line features for a fisheye rig.
 *
 * Formulation from the challenge winner (ACDC-VSLAM #1): ELSED segments,
 * endpoints lifted to bearings, each line represented by the NORMAL of its
 * great circle. A bearing b lies on the line iff n . b = 0, a residual that is
 * invariant to sliding along the line -- which is the aperture problem stated
 * honestly: a line carries ONE constraint, not two.
 *
 * Why this matters here: measured on floor_EG run_2 at the frames where
 * ORB-SLAM3 tracks 0-2 point features, ELSED still yields 643-993 segments per
 * camera and our prototype tracked 428 lines. The imagery is not empty; ORB's
 * descriptors just cannot use it.
 */
#ifndef LINEEXTRACTOR_H
#define LINEEXTRACTOR_H

#include <vector>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>

#include "CameraModels/GeometricCamera.h"

namespace ORB_SLAM3 {

/// One observed line segment in a frame.
struct LineObs {
    cv::Point2f p1, p2;        ///< pixel endpoints (as detected)
    Eigen::Vector3f b1u, b2u;  ///< UNIT bearings of the endpoints (camera frame)
    Eigen::Vector3f n;         ///< unit normal of the great circle, b1 x b2
    Eigen::Vector3f dir;       ///< in-plane direction, for the tracking gate
    float angLen;              ///< angular length [rad] -- NOT pixel length
    /// Local image scale [px/rad] where this segment sits: its own pixel
    /// length over its angular length. On a fisheye the pixels-per-radian
    /// varies strongly from centre to rim, so a single angular gate is not a
    /// single pixel gate -- LF-PGVIO measures its match residual (d_orth) in
    /// PIXELS for exactly this reason.
    float pxPerRad = 400.f;
    int cam;                   ///< 0 = front, 1 = rear (rig camera index)
    long mnLineId = -1;        ///< associated MapLine, -1 if none
    class MapLine* pML = nullptr;  ///< the landmark this track owns, if any

    // Anchor: the first observation of this track, and the camera pose there.
    // Triangulating against the anchor rather than the previous frame is what
    // gives the two interpretation planes enough angle to intersect.
    bool hasAnchor = false;
    /// 1-sigma noise of the ANCHOR observation's plane normal [rad]
    float anchorSigma = 0.0175f;
    /// How many consecutive frames this track has survived. PL-VINS refuses to
    /// triangulate a line until LINE_MIN_OBS (5) frames have seen it -- a
    /// 2-frame line carries no usable depth however good its parallax looks.
    int nSeen = 1;
    int anchorMapVersion = -1;   ///< Map::GetMapChangeIndex() when anchored
    Eigen::Vector3f nAnchor = Eigen::Vector3f::Zero();
    Eigen::Matrix3f RAnchor = Eigen::Matrix3f::Identity();
    Eigen::Vector3f tAnchor = Eigen::Vector3f::Zero();

    /// APPEARANCE. Band intensity profile across the edge: mean image
    /// intensity at {-5,-2,+2,+5} px along the local pixel normal, sampled
    /// along the segment, zero-mean/unit-norm normalised. Left/right is
    /// well-defined because the endpoint order is CANONICALISED (see
    /// CanonicalizeObs): two parallel edges of one bright strip -- geometric
    /// twins -- have opposite profiles. Geometry shortlists, THIS decides.
    float desc[4] = {0, 0, 0, 0};
    float contrast = 0.f;        ///< mean |across-edge gradient|, 0..255 scale
    int polarity = 0;            ///< sign of the across-edge gradient (+1/-1)
    bool hasDesc = false;
};

/// A persistent 2D line track: identity across frames, surviving brief
/// misses and failed triangulation. The 3D landmark (pML inside `last`)
/// attaches when geometry permits; the IDENTITY lives here.
struct LineTrack {
    long id = -1;
    LineObs last;                ///< most recent accepted observation
    /// last.n / last bearings rotated by the ACCUMULATED gyro rotation since
    /// the track was last seen -- where the edge SHOULD be now
    Eigen::Vector3f nPred = Eigen::Vector3f::Zero();
    Eigen::Vector3f b1Pred = Eigen::Vector3f::Zero();
    Eigen::Vector3f b2Pred = Eigen::Vector3f::Zero();
    int age = 0;                 ///< frames since birth
    int missed = 0;              ///< consecutive frames without a match
    int polSum = 0;              ///< running polarity vote (history)
    float descEma[4] = {0,0,0,0};///< EMA of the band profile (history)
    bool emaInit = false;
};

class LineExtractor {
public:
    LineExtractor(float minAngLenDeg = 1.5f, float maxAngLenDeg = 40.0f,
                  int gradThresh = 30, int minLenPx = 15);

    /// Detect + lift. Segments whose endpoints do not unproject validly are
    /// dropped: an unchecked unprojection is exactly the bug that gave OKVIS2
    /// false cross-camera overlap.
    std::vector<LineObs> Extract(const cv::Mat& imGray, GeometricCamera* pCam,
                                 int camIdx) const;

    /// Self-occlusion mask per camera. NONZERO == MASKED (ignore), the same
    /// convention as everywhere else here. Features on the rig's own hardware
    /// are worse than none: the hardware is static in the camera frame, so it
    /// reports "no motion" while the rig moves.
    void SetMask(int camIdx, const cv::Mat& mask);
    cv::Mat mMask[2];

    /// Match `cur` against `prev` with the winner's three gates:
    /// normal alignment, in-plane direction, angular-length consistency.
    /// Comparisons are sign-free -- a line has no orientation.
    /// Returns cur-index -> prev-index, or -1.
    /// Collapse collinear fragments of one edge into a single circle.
    std::vector<LineObs> MergeGreatCircles(const std::vector<LineObs>& in) const;

    std::vector<int> Match(const std::vector<LineObs>& cur,
                           const std::vector<LineObs>& prev) const;

    /// Canonical endpoint order: n = b1 x b2 with n_z > 0 (n_x tie-break).
    /// Detector endpoint order is arbitrary; the band descriptor's left/right
    /// only means anything once the traversal direction is fixed.
    static void CanonicalizeObs(LineObs& o);
    /// Band profile + polarity from the image, sampled along the (possibly
    /// curved) pixel polyline of the observation. Fills desc/contrast/polarity.
    static void ComputeBandDescriptor(const cv::Mat& img,
                                      const std::vector<cv::Point2f>& poly,
                                      LineObs& o);
    /// L2 distance between normalised band profiles (2.0 = worst).
    static float DescDist(const float a[4], const float b[4]);
    /// 1-sigma noise of this observation's plane NORMAL [rad]:
    /// sqrt(2)*sigma_px / segment_pixel_length. A direction claim must clear
    /// MULTIPLES of this or it is noise -- measured: horizontal directions at
    /// isotropic chance (11%) while every residual audit passed, because the
    /// plane residual is blind to direction inside a thin sheaf.
    static float NormalSigma(const LineObs& o) {
        const float lenPx = o.angLen * o.pxPerRad;
        return 1.41421356f / (lenPx > 5.f ? lenPx : 5.f);
    }

    float mMinAngLen, mMaxAngLen;   ///< [rad]
    float mGateNormal, mGateDir;    ///< [rad]
    float mGateLenRatio;
    /// Minimum fraction of the SHORTER segment that must overlap the other,
    /// measured as arc along their shared great circle (LF-PGVIO uses 0.5).
    float mGateOverlap = 0.5f;
    /// Max orthogonal distance of an endpoint from the other segment's great
    /// circle, in PIXELS (LF-PGVIO: d_orth < 5 px).
    float mGateOrthPx = 5.0f;
    /// Merge collinear fragments into one great-circle observation before
    /// matching (Lines.mergeCircles). Normals within mMergeNormalRad are the
    /// same circle; fragments are joined across a gap of at most
    /// mMergeMaxGap x the longer piece.
    bool  mbMergeCircles = true;
    float mMergeNormalRad = 0.5f * float(M_PI) / 180.f;   // 0.5 deg
    float mMergeMaxGap = 1.0f;
    mutable GeometricCamera* mpCamForMerge = nullptr;
    int mGradThresh, mMinLenPx;
};

/**
 * Persistent line tracker, one per lens:
 *   gyro-predict -> loose geometric shortlist -> appearance decides
 *   -> ambiguity checks -> per-fragment assignment.
 *
 * Identity lives HERE (track ids survive brief misses and failed
 * triangulation); the 3D landmark rides inside the track's LineObs and
 * attaches when geometry permits. Fragments are friends: several current
 * fragments may bind to ONE track (that is what a broken edge is); ratio
 * ambiguity is only measured between COMPETING track identities.
 *
 * Two-phase per frame:
 *   Match(cur, dR)  predicts every live track by the gyro rotation, matches,
 *                   copies track identity (pML/anchor/nSeen) INTO the matched
 *                   cur observations, returns track index per cur (-1 = new).
 *   Commit(cur,asg) stores the frame's FINAL observations back into the
 *                   tracks (Tracking mutates them in between: triangulation,
 *                   landmark binding), spawns tracks for the unmatched.
 */
class LineTracker {
public:
    /// dR_cur_prev: rotation taking PREV-frame camera coords to CUR-frame
    /// camera coords (bias-corrected gyro; identity if unavailable).
    std::vector<int> Match(std::vector<LineObs>& cur,
                           const Eigen::Matrix3f& dR_cur_prev);
    void Commit(const std::vector<LineObs>& cur, const std::vector<int>& asg);
    void Reset() { mTracks.clear(); }
    const std::vector<LineTrack>& Tracks() const { return mTracks; }

    // gates: LOOSE -- they only shortlist, appearance decides
    float mGateNormalRad = 3.0f * 3.14159265f / 180.f;
    float mGateOrthPx = 12.f;
    float mGateOverlap = 0.3f;
    // decision thresholds
    float mDescAbsMax = 0.55f;   ///< absolute descriptor quality (not just ratio)
    float mDescRatioMax = 0.8f;  ///< distinctiveness vs best COMPETING track
    float mMinContrast = 8.f;    ///< below this the polarity/desc is noise
    int mMaxMissed = 5;          ///< track survives this many blind frames

    /// Per-frame rejection counters -- which rule killed each candidate.
    struct Stats {
        long nCand = 0, killNormal = 0, killOrth = 0, killOverlap = 0,
             killPolarity = 0, killAbs = 0, killRatio = 0, killHistory = 0,
             matched = 0, fresh = 0, dropped = 0;
    } mStats;

private:
    std::vector<LineTrack> mTracks;
    long mNextId = 0;
};

}  // namespace ORB_SLAM3
#endif

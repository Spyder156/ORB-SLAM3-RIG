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
    int cam;                   ///< 0 = front, 1 = rear (rig camera index)
    long mnLineId = -1;        ///< associated MapLine, -1 if none
    class MapLine* pML = nullptr;  ///< the landmark this track owns, if any

    // Anchor: the first observation of this track, and the camera pose there.
    // Triangulating against the anchor rather than the previous frame is what
    // gives the two interpretation planes enough angle to intersect.
    bool hasAnchor = false;
    int anchorMapVersion = -1;   ///< Map::GetMapChangeIndex() when anchored
    Eigen::Vector3f nAnchor = Eigen::Vector3f::Zero();
    Eigen::Matrix3f RAnchor = Eigen::Matrix3f::Identity();
    Eigen::Vector3f tAnchor = Eigen::Vector3f::Zero();
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
    std::vector<int> Match(const std::vector<LineObs>& cur,
                           const std::vector<LineObs>& prev) const;

    float mMinAngLen, mMaxAngLen;   ///< [rad]
    float mGateNormal, mGateDir;    ///< [rad]
    float mGateLenRatio;
    int mGradThresh, mMinLenPx;
};

}  // namespace ORB_SLAM3
#endif

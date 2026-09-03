/**
 * Rig.h -- non-overlapping multi-camera rig for ORB-SLAM3.
 *
 * ORB-SLAM3 ships two-camera plumbing (mpCamera2 / mTlr / Stereo.T_c1_c2) but it
 * is gated behind the STEREO sensor types and assumes the two views OVERLAP.
 * Our rig is back-to-back: the cameras share no field of view at any instant.
 * This module holds the rig as a first-class object so the rest of the system can
 * reason about it without inheriting the stereo assumptions.
 *
 * Extrinsics come from Step 0 (COLMAP rig-constrained BA + IMU metric scale),
 * NOT from an assumed 180 deg / zero baseline.
 *
 * Refinement policy (measured, not guessed): with non-overlapping cameras the
 * rig ROTATION is well observed but the TRANSLATION is not -- in Step 0, BA
 * improved the angle and simultaneously degraded the baseline by 24%. So the
 * default is: refine rotation, hold translation.
 */
#ifndef RIG_H
#define RIG_H

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>

namespace ORB_SLAM3 {

class Rig {
public:
    Rig() = default;

    /// Load from the settings yaml. Keys (all optional; absent => rig disabled):
    ///   Rig.enabled            : 1/0
    ///   Rig.T_c0_c1            : 4x4, camera1 -> camera0
    ///   Rig.refine_rotation    : 1/0  (default 1)
    ///   Rig.refine_translation : 1/0  (default 0, see header note)
    bool LoadFromSettings(const std::string& settingsPath);

    bool IsEnabled() const { return mbEnabled; }
    int NumCameras() const { return mbEnabled ? 2 : 1; }

    /// camera1 -> camera0 (cam0 is the rig reference sensor)
    const Sophus::SE3f& T_c0_c1() const { return mT_c0_c1; }
    Sophus::SE3f T_c1_c0() const { return mT_c0_c1.inverse(); }

    /// Angle between the two optical axes, degrees. 180 == ideal back-to-back.
    float InterCameraAngleDeg() const;
    /// Distance between the two optical centres, metres.
    float BaselineMetres() const;

    /// True when the two cameras share NO field of view. ORB-SLAM3's two-camera
    /// path is written for stereo fisheye and assumes an overlap region
    /// (KannalaBrandt8::mvLappingArea); on a back-to-back rig that region does
    /// not exist and cross-camera matching produces only false matches.
    bool NonOverlapping() const { return mbNonOverlapping; }

    // ---- Experiment B: zero-baseline start, refined online -----------------
    // See SLAM/patches/orbslam3_rigB/README.md.
    //
    // The metric baseline (4.01 cm) is meaningless in a monocular map whose
    // units are arbitrary. A ZERO baseline is scale-invariant, so the rig can
    // be active from frame 1 with only its (scale-free) rotation, and the
    // translation is released later once the IMU has pinned the scale.

    /// Rotation is scale-free -- always valid, always used.
    const Eigen::Matrix3f& R_c0_c1() const { return mR_c0_c1; }

    /// Baseline to USE right now, in MAP units. Zero during phase 1.
    Eigen::Vector3f BaselineMapUnits() const {
        return mbBaselineReleased ? mBaselineMap : Eigen::Vector3f::Zero();
    }
    bool BaselineReleased() const { return mbBaselineReleased; }

    /// Called once the map is metric (IMU scale converged). mapScale converts
    /// MAP units -> metres. The prior target is the Step 0 metric baseline
    /// expressed in map units.
    void ReleaseBaseline(float mapScale);

    /// Soft-prior target, map units. NOT a hard constraint: with a
    /// non-overlapping rig the baseline is weakly observable and a free
    /// parameter wanders (Step 0: rig BA degraded it 24%).
    const Eigen::Vector3f& BaselinePrior() const { return mBaselinePrior; }

    /// Log the current estimate in METRES. Success = converges to ~4.01 cm.
    void LogBaseline(float mapScale, long keyframeId) const;

    bool RefineRotation() const { return mbRefineRotation; }
    bool RefineTranslation() const { return mbRefineTranslation; }

    /// One-line summary for the log. Values are deliberately NOT printed by the
    /// stage hooks -- only by this, once, at load.
    void PrintSummary() const;

    /// Stage markers. These exist so the rig's path through the system is
    /// traceable in a log without dumping numbers on every frame.
    static void Stage(const char* where, const char* what);

    /// Same, but prints only the FIRST time for a given message -- for hooks on
    /// the per-frame path, which would otherwise flood the log.
    static void StageOnce(const char* where, const char* what);

    /// Frame/KeyFrame are constructed deep in the pipeline with no handle on the
    /// System that owns the Rig. Rather than thread a pointer through every
    /// constructor, the loaded rig publishes this one flag statically. It is set
    /// once, before any thread starts, and read-only thereafter.
    static bool NonOverlappingGlobal();
    static void PublishGlobals(bool nonOverlapping);

private:
    bool mbEnabled = false;
    bool mbRefineRotation = true;
    bool mbNonOverlapping = true;
    bool mbBaselineReleased = false;
    Eigen::Matrix3f mR_c0_c1 = Eigen::Matrix3f::Identity();
    Eigen::Vector3f mBaselineMap = Eigen::Vector3f::Zero();    // live estimate
    Eigen::Vector3f mBaselinePrior = Eigen::Vector3f::Zero();  // soft target
    bool mbRefineTranslation = false;
    Sophus::SE3f mT_c0_c1;
};

}  // namespace ORB_SLAM3

#endif

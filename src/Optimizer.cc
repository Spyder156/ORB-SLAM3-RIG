/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include "Optimizer.h"
#include "MapLine.h"


#include <complex>

#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "Thirdparty/g2o/g2o/core/sparse_block_matrix.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "G2oTypes.h"
#include "Converter.h"

#include<mutex>

#include "OptimizableTypes.h"


namespace ORB_SLAM3
{

/// Withhold point edges from the inertial pose optimizer, so IMU + lines alone
/// fix the pose. Set from Lines.only in the settings file.
bool Optimizer::bLinesOnly = false;

bool sortByVal(const pair<MapPoint*, int> &a, const pair<MapPoint*, int> &b)
{
    return (a.second < b.second);
}

void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
}


void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    Map* pMap = vpKFs[0]->GetMap();

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    const int nExpectedSize = (vpKFs.size())*vpMP.size();

    vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody*> vpEdgesBody;
    vpEdgesBody.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFBody;
    vpEdgeKFBody.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeBody;
    vpMapPointEdgeBody.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);


    // Set KeyFrame vertices

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKF->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId==pMap->GetInitKFid());
        optimizer.addVertex(vSE3);
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    for(size_t i=0; i<vpMP.size(); i++)
    {
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        const int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

       const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {
            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid)
                continue;
            if(optimizer.vertex(id) == NULL || optimizer.vertex(pKF->mnId) == NULL)
                continue;
            nEdges++;

            const int leftIndex = get<0>(mit->second);

            if(leftIndex != -1 && pKF->mvuRight[get<0>(mit->second)]<0)
            {
                const cv::KeyPoint &kpUn = pKF->mvKeysUn[leftIndex];

                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->pCamera = pKF->mpCamera;

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKF);
                vpMapPointEdgeMono.push_back(pMP);
            }
            else if(leftIndex != -1 && pKF->mvuRight[leftIndex] >= 0) //Stereo observation
            {
                const cv::KeyPoint &kpUn = pKF->mvKeysUn[leftIndex];

                Eigen::Matrix<double,3,1> obs;
                const float kp_ur = pKF->mvuRight[get<0>(mit->second)];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vpEdgeKFStereo.push_back(pKF);
                vpMapPointEdgeStereo.push_back(pMP);
            }

            if(pKF->mpCamera2){
                // rightIndex is the CONCATENATED keypoint index (>= NLeft for
                // any rear observation); mvKeysRight is rear-only. Checking the
                // concatenated index against the rear-only size before
                // subtracting NLeft rejected EVERY rear observation whenever
                // NLeft > rear count -- rear cameras silently absent from BA.
                int rightIndex = get<1>(mit->second);

                if(rightIndex != -1){
                    rightIndex -= pKF->NLeft;
                    if(rightIndex < 0 || rightIndex >= (int)pKF->mvKeysRight.size())
                        continue;

                    Eigen::Matrix<double,2,1> obs;
                    cv::KeyPoint kp = pKF->mvKeysRight[rightIndex];
                    obs << kp.pt.x, kp.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->mvInvLevelSigma2[kp.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);

                    Sophus::SE3f Trl = pKF-> GetRelativePoseTrl();
                    e->mTrl = g2o::SE3Quat(Trl.unit_quaternion().cast<double>(), Trl.translation().cast<double>());

                    e->pCamera = pKF->mpCamera2;

                    optimizer.addEdge(e);
                    vpEdgesBody.push_back(e);
                    vpEdgeKFBody.push_back(pKF);
                    vpMapPointEdgeBody.push_back(pMP);
                }
            }
        }



        if(nEdges==0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }

    // Per-camera edge audit: rear edges were silently ZERO before the NLeft
    // bounds-order fix; this line is the regression witness.
    std::cout << "[AUDIT] GBA edges: front " << vpEdgesMono.size()
              << " rear " << vpEdgesBody.size() << std::endl;

    // Optimize!
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);
    Verbose::PrintMess("BA: End of the optimization", Verbose::VERBOSITY_NORMAL);

    // Recover optimized data
    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));

        g2o::SE3Quat SE3quat = vSE3->estimate();
        if(nLoopKF==pMap->GetOriginKF()->mnId)
        {
            pKF->SetPose(Sophus::SE3f(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>()));
        }
        else
        {
            pKF->mTcwGBA = Sophus::SE3d(SE3quat.rotation(),SE3quat.translation()).cast<float>();
            pKF->mnBAGlobalForKF = nLoopKF;

            Sophus::SE3f mTwc = pKF->GetPoseInverse();
            Sophus::SE3f mTcGBA_c = pKF->mTcwGBA * mTwc;
            Eigen::Vector3f vector_dist =  mTcGBA_c.translation();
            double dist = vector_dist.norm();
            if(dist > 1)
            {
                int numMonoBadPoints = 0, numMonoOptPoints = 0;
                int numStereoBadPoints = 0, numStereoOptPoints = 0;
                vector<MapPoint*> vpMonoMPsOpt, vpStereoMPsOpt;

                for(size_t i2=0, iend=vpEdgesMono.size(); i2<iend;i2++)
                {
                    ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i2];
                    MapPoint* pMP = vpMapPointEdgeMono[i2];
                    KeyFrame* pKFedge = vpEdgeKFMono[i2];

                    if(pKF != pKFedge)
                    {
                        continue;
                    }

                    if(pMP->isBad())
                        continue;

                    if(e->chi2()>5.991 || !e->isDepthPositive())
                    {
                        numMonoBadPoints++;

                    }
                    else
                    {
                        numMonoOptPoints++;
                        vpMonoMPsOpt.push_back(pMP);
                    }

                }

                for(size_t i2=0, iend=vpEdgesStereo.size(); i2<iend;i2++)
                {
                    g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i2];
                    MapPoint* pMP = vpMapPointEdgeStereo[i2];
                    KeyFrame* pKFedge = vpEdgeKFMono[i2];

                    if(pKF != pKFedge)
                    {
                        continue;
                    }

                    if(pMP->isBad())
                        continue;

                    if(e->chi2()>7.815 || !e->isDepthPositive())
                    {
                        numStereoBadPoints++;
                    }
                    else
                    {
                        numStereoOptPoints++;
                        vpStereoMPsOpt.push_back(pMP);
                    }
                }
            }
        }
    }

    //Points
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];

        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

        if(nLoopKF==pMap->GetOriginKF()->mnId)
        {
            pMP->SetWorldPos(vPoint->estimate().cast<float>());
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA = vPoint->estimate().cast<float>();
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }
}

void Optimizer::FullInertialBA(Map *pMap, int its, const bool bFixLocal, const long unsigned int nLoopId, bool *pbStopFlag, bool bInit, float priorG, float priorA, Eigen::VectorXd *vSingVal, bool *bHess)
{
    // per-camera edge audit counters (printed + reset before optimize)
    static long skFIBAFrontEdges = 0, skFIBARearEdges = 0;
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setUserLambdaInit(1e-5);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    int nNonFixed = 0;

    // Set KeyFrame vertices
    KeyFrame* pIncKF;
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        pIncKF=pKFi;
        bool bFixed = false;
        if(bFixLocal)
        {
            bFixed = (pKFi->mnBALocalForKF>=(maxKFid-1)) || (pKFi->mnBAFixedForKF>=(maxKFid-1));
            if(!bFixed)
                nNonFixed++;
            VP->setFixed(bFixed);
        }
        optimizer.addVertex(VP);

        if(pKFi->bImu)
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->mnId)+1);
            VV->setFixed(bFixed);
            optimizer.addVertex(VV);
            if (!bInit)
            {
                VertexGyroBias* VG = new VertexGyroBias(pKFi);
                VG->setId(maxKFid+3*(pKFi->mnId)+2);
                VG->setFixed(bFixed);
                optimizer.addVertex(VG);
                VertexAccBias* VA = new VertexAccBias(pKFi);
                VA->setId(maxKFid+3*(pKFi->mnId)+3);
                VA->setFixed(bFixed);
                optimizer.addVertex(VA);
            }
        }
    }

    if (bInit)
    {
        VertexGyroBias* VG = new VertexGyroBias(pIncKF);
        VG->setId(4*maxKFid+2);
        VG->setFixed(false);
        optimizer.addVertex(VG);
        VertexAccBias* VA = new VertexAccBias(pIncKF);
        VA->setId(4*maxKFid+3);
        VA->setFixed(false);
        optimizer.addVertex(VA);
    }

    if(bFixLocal)
    {
        if(nNonFixed<3)
            return;
    }

    // IMU links
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        if(!pKFi->mPrevKF)
        {
            Verbose::PrintMess("NOT INERTIAL LINK TO PREVIOUS FRAME!", Verbose::VERBOSITY_NORMAL);
            continue;
        }

        if(pKFi->mPrevKF && pKFi->mnId<=maxKFid)
        {
            if(pKFi->isBad() || pKFi->mPrevKF->mnId>maxKFid)
                continue;
            if(pKFi->bImu && pKFi->mPrevKF->bImu)
            {
                pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
                g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
                g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+1);

                g2o::HyperGraph::Vertex* VG1;
                g2o::HyperGraph::Vertex* VA1;
                g2o::HyperGraph::Vertex* VG2;
                g2o::HyperGraph::Vertex* VA2;
                if (!bInit)
                {
                    VG1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+2);
                    VA1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+3);
                    VG2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+2);
                    VA2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+3);
                }
                else
                {
                    VG1 = optimizer.vertex(4*maxKFid+2);
                    VA1 = optimizer.vertex(4*maxKFid+3);
                }

                g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
                g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+1);

                if (!bInit)
                {
                    if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
                    {
                        cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                        continue;
                    }
                }
                else
                {
                    if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2)
                    {
                        cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<endl;
                        continue;
                    }
                }

                EdgeInertial* ei = new EdgeInertial(pKFi->mpImuPreintegrated);
                ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
                ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
                ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
                ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
                ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
                ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

                g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
                ei->setRobustKernel(rki);
                rki->setDelta(sqrt(16.92));

                optimizer.addEdge(ei);

                if (!bInit)
                {
                    EdgeGyroRW* egr= new EdgeGyroRW();
                    egr->setVertex(0,VG1);
                    egr->setVertex(1,VG2);
                    Eigen::Matrix3d InfoG = pKFi->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
                    egr->setInformation(InfoG);
                    egr->computeError();
                    optimizer.addEdge(egr);

                    EdgeAccRW* ear = new EdgeAccRW();
                    ear->setVertex(0,VA1);
                    ear->setVertex(1,VA2);
                    Eigen::Matrix3d InfoA = pKFi->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
                    ear->setInformation(InfoA);
                    ear->computeError();
                    optimizer.addEdge(ear);
                }
            }
            else
                cout << pKFi->mnId << " or " << pKFi->mPrevKF->mnId << " no imu" << endl;
        }
    }

    if (bInit)
    {
        g2o::HyperGraph::Vertex* VG = optimizer.vertex(4*maxKFid+2);
        g2o::HyperGraph::Vertex* VA = optimizer.vertex(4*maxKFid+3);

        // Add prior to comon biases
        Eigen::Vector3f bprior;
        bprior.setZero();

        EdgePriorAcc* epa = new EdgePriorAcc(bprior);
        epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
        double infoPriorA = priorA; //
        epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
        optimizer.addEdge(epa);

        EdgePriorGyro* epg = new EdgePriorGyro(bprior);
        epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
        double infoPriorG = priorG; //
        epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
        optimizer.addEdge(epg);
    }

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    const unsigned long iniMPid = maxKFid*5;

    vector<bool> vbNotIncludedMP(vpMPs.size(),false);

    for(size_t i=0; i<vpMPs.size(); i++)
    {
        MapPoint* pMP = vpMPs[i];
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        unsigned long id = pMP->mnId+iniMPid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();


        bool bAllFixed = true;

        //Set edges
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnId>maxKFid)
                continue;

            if(!pKFi->isBad())
            {
                const int leftIndex = get<0>(mit->second);
                cv::KeyPoint kpUn;

                if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]<0) // Monocular observation
                {
                    kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMono* e = new EdgeMono(0);

                    g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId));
                    if(bAllFixed)
                        if(!VP->fixed())
                            bAllFixed=false;

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, VP);
                    e->setMeasurement(obs);
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];

                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);
                    skFIBAFrontEdges++;
                }
                else if(leftIndex != -1 && pKFi->mvuRight[leftIndex] >= 0) // stereo observation
                {
                    kpUn = pKFi->mvKeysUn[leftIndex];
                    const float kp_ur = pKFi->mvuRight[leftIndex];
                    Eigen::Matrix<double,3,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereo* e = new EdgeStereo(0);

                    g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId));
                    if(bAllFixed)
                        if(!VP->fixed())
                            bAllFixed=false;

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, VP);
                    e->setMeasurement(obs);
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];

                    e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);
                }

                if(pKFi->mpCamera2){ // Monocular right observation
                    // Same NLeft-order bug as BundleAdjustment: the bound must
                    // be checked on the REAR-LOCAL index, not the concatenated
                    // one, or every rear observation is dropped from the very
                    // BA that initialisation runs (LocalMapping calls this).
                    int rightIndex = get<1>(mit->second);

                    if(rightIndex != -1){
                        rightIndex -= pKFi->NLeft;
                        if(rightIndex < 0 || rightIndex >= (int)pKFi->mvKeysRight.size())
                            continue;

                        Eigen::Matrix<double,2,1> obs;
                        kpUn = pKFi->mvKeysRight[rightIndex];
                        obs << kpUn.pt.x, kpUn.pt.y;

                        EdgeMono *e = new EdgeMono(1);

                        g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId));
                        if(bAllFixed)
                            if(!VP->fixed())
                                bAllFixed=false;

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, VP);
                        e->setMeasurement(obs);
                        const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);
                        skFIBARearEdges++;
                    }
                }
            }
        }

        if(bAllFixed)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
    }

    // Per-camera edge audit (regression witness for the NLeft bounds fix:
    // rear was silently 0 before it).
    std::cout << "[AUDIT] FIBA edges: front " << skFIBAFrontEdges
              << " rear " << skFIBARearEdges << std::endl;
    skFIBAFrontEdges = skFIBARearEdges = 0;

    if(pbStopFlag)
        if(*pbStopFlag)
            return;


    optimizer.initializeOptimization();
    optimizer.optimize(its);


    // Recover optimized data
    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;
        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        if(nLoopId==0)
        {
            Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
            pKFi->SetPose(Tcw);
        }
        else
        {
            pKFi->mTcwGBA = Sophus::SE3f(VP->estimate().Rcw[0].cast<float>(),VP->estimate().tcw[0].cast<float>());
            pKFi->mnBAGlobalForKF = nLoopId;

        }
        if(pKFi->bImu)
        {
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
            if(nLoopId==0)
            {
                pKFi->SetVelocity(VV->estimate().cast<float>());
            }
            else
            {
                pKFi->mVwbGBA = VV->estimate().cast<float>();
            }

            VertexGyroBias* VG;
            VertexAccBias* VA;
            if (!bInit)
            {
                VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
                VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
            }
            else
            {
                VG = static_cast<VertexGyroBias*>(optimizer.vertex(4*maxKFid+2));
                VA = static_cast<VertexAccBias*>(optimizer.vertex(4*maxKFid+3));
            }

            Vector6d vb;
            vb << VG->estimate(), VA->estimate();
            IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);
            if(nLoopId==0)
            {
                pKFi->SetNewBias(b);
            }
            else
            {
                pKFi->mBiasGBA = b;
            }
        }
    }

    //Points
    for(size_t i=0; i<vpMPs.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMPs[i];
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+iniMPid+1));

        if(nLoopId==0)
        {
            pMP->SetWorldPos(vPoint->estimate().cast<float>());
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA = vPoint->estimate().cast<float>();
            pMP->mnBAGlobalForKF = nLoopId;
        }

    }

    pMap->IncreaseChangeIndex();
}


int Optimizer::PoseOptimization(Frame *pFrame)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    Sophus::SE3<float> Tcw = pFrame->GetPose();
    vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *> vpEdgesMono_FHR;
    vector<size_t> vnIndexEdgeMono, vnIndexEdgeRight;
    vpEdgesMono.reserve(N);
    vpEdgesMono_FHR.reserve(N);
    vnIndexEdgeMono.reserve(N);
    vnIndexEdgeRight.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP)
        {
            //Conventional SLAM
            if(!pFrame->mpCamera2){
                // Monocular observation
                if(pFrame->mvuRight[i]<0)
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double,2,1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    obs << kpUn.pt.x, kpUn.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->pCamera = pFrame->mpCamera;
                    e->Xw = pMP->GetWorldPos().cast<double>();

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                else  // Stereo observation
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double,3,1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    const float &kp_ur = pFrame->mvuRight[i];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaStereo);

                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    e->bf = pFrame->mbf;
                    e->Xw = pMP->GetWorldPos().cast<double>();

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }
            }
            //SLAM with respect a rigid body
            else{
                nInitialCorrespondences++;

                cv::KeyPoint kpUn;

                if (i < pFrame->Nleft) {    //Left camera observation
                    kpUn = pFrame->mvKeys[i];

                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->pCamera = pFrame->mpCamera;
                    e->Xw = pMP->GetWorldPos().cast<double>();

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                else {
                    kpUn = pFrame->mvKeysRight[i - pFrame->Nleft];

                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    pFrame->mvbOutlier[i] = false;

                    ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->pCamera = pFrame->mpCamera2;
                    e->Xw = pMP->GetWorldPos().cast<double>();

                    e->mTrl = g2o::SE3Quat(pFrame->GetRelativePoseTrl().unit_quaternion().cast<double>(), pFrame->GetRelativePoseTrl().translation().cast<double>());

                    optimizer.addEdge(e);

                    vpEdgesMono_FHR.push_back(e);
                    vnIndexEdgeRight.push_back(i);
                }
            }
        }
    }
    }

    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};    

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {
        Tcw = pFrame->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));

        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {                
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesMono_FHR.size(); i<iend; i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody* e = vpEdgesMono_FHR[i];

            const size_t idx = vnIndexEdgeRight[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {                
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        if(optimizer.edges().size()<10)
            break;
    }    

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    Sophus::SE3<float> pose(SE3quat_recov.rotation().cast<float>(),
            SE3quat_recov.translation().cast<float>());
    pFrame->SetPose(pose);

    return nInitialCorrespondences-nBad;
}

void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap, int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges)
{
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    Map* pCurrentMap = pKF->GetMap();

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    num_fixedKF = 0;
    list<MapPoint*> lLocalMapPoints;
    set<MapPoint*> sNumObsMP;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        if(pKFi->mnId==pMap->GetInitKFid())
        {
            num_fixedKF = 1;
        }
        vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad() && pMP->GetMap() == pCurrentMap)
                {

                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
                }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,tuple<int,int>> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId )
            {                
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }
    num_fixedKF = lFixedCameras.size() + num_fixedKF;


    if(num_fixedKF == 0)
    {
        Verbose::PrintMess("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted", Verbose::VERBOSITY_NORMAL);
        return;
    }

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    if (pMap->IsInertial())
        solver->setUserLambdaInit(100.0);

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // DEBUG LBA
    pCurrentMap->msOptKFs.clear();
    pCurrentMap->msFixedKFs.clear();

    // Set Local KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId==pMap->GetInitKFid());
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
        // DEBUG LBA
        pCurrentMap->msOptKFs.insert(pKFi->mnId);
    }
    num_OptKF = lLocalKeyFrames.size();

    // Set Fixed KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
        // DEBUG LBA
        pCurrentMap->msFixedKFs.insert(pKFi->mnId);
    }

    // Set MapPoint vertices
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

    vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody*> vpEdgesBody;
    vpEdgesBody.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFBody;
    vpEdgeKFBody.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeBody;
    vpMapPointEdgeBody.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    int nPoints = 0;

    int nEdges = 0;

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        nPoints++;

        const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

        //Set edges
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            {
                const int leftIndex = get<0>(mit->second);

                // Monocular observation
                if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]<0)
                {
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->pCamera = pKFi->mpCamera;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);

                    nEdges++;
                }
                else if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]>=0)// Stereo observation
                {
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKFi->mvuRight[get<0>(mit->second)];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);

                    nEdges++;
                }

                if(pKFi->mpCamera2){
                    int rightIndex = get<1>(mit->second);

                    if(rightIndex != -1 ){
                        rightIndex -= pKFi->NLeft;

                        Eigen::Matrix<double,2,1> obs;
                        cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                        obs << kp.pt.x, kp.pt.y;

                        ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kp.octave];
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        Sophus::SE3f Trl = pKFi-> GetRelativePoseTrl();
                        e->mTrl = g2o::SE3Quat(Trl.unit_quaternion().cast<double>(), Trl.translation().cast<double>());

                        e->pCamera = pKFi->mpCamera2;

                        optimizer.addEdge(e);
                        vpEdgesBody.push_back(e);
                        vpEdgeKFBody.push_back(pKFi);
                        vpMapPointEdgeBody.push_back(pMP);

                        nEdges++;
                    }
                }
            }
        }
    }
    num_edges = nEdges;

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesBody.size()+vpEdgesStereo.size());

    // Check inlier observations       
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    for(size_t i=0, iend=vpEdgesBody.size(); i<iend;i++)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZToBody* e = vpEdgesBody[i];
        MapPoint* pMP = vpMapPointEdgeBody[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFBody[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }


    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data
    //Keyframes
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());
        pKFi->SetPose(Tiw);
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();
    }

    pMap->IncreaseChangeIndex();
}


void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
{   
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    vector<Eigen::Vector3d> vZvectors(nMaxKFid+1); // For debugging
    Eigen::Vector3d z_vec;
    z_vec << 0.0, 0.0, 1.0;

    const int minFeat = 100;

    // Set KeyFrame vertices
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->mnId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        if(it!=CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else
        {
            Sophus::SE3d Tcw = pKF->GetPose().cast<double>();
            g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        if(pKF->mnId==pMap->GetInitKFid())
            VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);
        vZvectors[nIDi]=vScw[nIDi].rotation()*z_vec; // For debugging

        vpVertices[nIDi]=VSim3;
    }


    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    // Set Loop edges
    int count_loop = 0;
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];
        const g2o::Sim3 Swi = Siw.inverse();

        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);
            count_loop++;
            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // Set normal edges
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if(iti!=NonCorrectedSim3.end())
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKF = pKF->GetParent();

        // Spanning tree edge
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if(itj!=NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);
            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(pLKF->mnId<pKF->mnId)
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if(itl!=NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) /*&& !sLoopEdges.count(pKFn)*/)
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if(itn!=NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }

        // Inertial edges if inertial
        if(pKF->bImu && pKF->mPrevKF)
        {
            g2o::Sim3 Spw;
            LoopClosing::KeyFrameAndPose::const_iterator itp = NonCorrectedSim3.find(pKF->mPrevKF);
            if(itp!=NonCorrectedSim3.end())
                Spw = itp->second;
            else
                Spw = vScw[pKF->mPrevKF->mnId];

            g2o::Sim3 Spi = Spw * Swi;
            g2o::EdgeSim3* ep = new g2o::EdgeSim3();
            ep->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mPrevKF->mnId)));
            ep->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            ep->setMeasurement(Spi);
            ep->information() = matLambda;
            optimizer.addEdge(ep);
        }
    }


    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    optimizer.optimize(20);
    optimizer.computeActiveErrors();
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        double s = CorrectedSiw.scale();

        Sophus::SE3f Tiw(CorrectedSiw.rotation().cast<float>(), CorrectedSiw.translation().cast<float>() / s);
        pKFi->SetPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;
        if(pMP->mnCorrectedByKF==pCurKF->mnId)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }


        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        Eigen::Matrix<double,3,1> eigP3Dw = pMP->GetWorldPos().cast<double>();
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
        pMP->SetWorldPos(eigCorrectedP3Dw.cast<float>());

        pMP->UpdateNormalAndDepth();
    }

    // TODO Check this changeindex
    pMap->IncreaseChangeIndex();
}

void Optimizer::OptimizeEssentialGraph(KeyFrame* pCurKF, vector<KeyFrame*> &vpFixedKFs, vector<KeyFrame*> &vpFixedCorrectedKFs,
                                       vector<KeyFrame*> &vpNonFixedKFs, vector<MapPoint*> &vpNonCorrectedMPs)
{
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpFixedKFs.size()) + " KFs fixed in the merged map", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpFixedCorrectedKFs.size()) + " KFs fixed in the old map", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpNonFixedKFs.size()) + " KFs non-fixed in the merged map", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpNonCorrectedMPs.size()) + " MPs non-corrected in the merged map", Verbose::VERBOSITY_DEBUG);

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    Map* pMap = pCurKF->GetMap();
    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    vector<bool> vpGoodPose(nMaxKFid+1);
    vector<bool> vpBadPose(nMaxKFid+1);

    const int minFeat = 100;

    for(KeyFrame* pKFi : vpFixedKFs)
    {
        if(pKFi->isBad())
            continue;

        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKFi->mnId;

        Sophus::SE3d Tcw = pKFi->GetPose().cast<double>();
        g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);

        vCorrectedSwc[nIDi]=Siw.inverse();
        VSim3->setEstimate(Siw);

        VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = true;

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;

        vpGoodPose[nIDi] = true;
        vpBadPose[nIDi] = false;
    }
    Verbose::PrintMess("Opt_Essential: vpFixedKFs loaded", Verbose::VERBOSITY_DEBUG);

    set<unsigned long> sIdKF;
    for(KeyFrame* pKFi : vpFixedCorrectedKFs)
    {
        if(pKFi->isBad())
            continue;

        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKFi->mnId;

        Sophus::SE3d Tcw = pKFi->GetPose().cast<double>();
        g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);

        vCorrectedSwc[nIDi]=Siw.inverse();
        VSim3->setEstimate(Siw);

        Sophus::SE3d Tcw_bef = pKFi->mTcwBefMerge.cast<double>();
        vScw[nIDi] = g2o::Sim3(Tcw_bef.unit_quaternion(),Tcw_bef.translation(),1.0);

        VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;

        sIdKF.insert(nIDi);

        vpGoodPose[nIDi] = true;
        vpBadPose[nIDi] = true;
    }

    for(KeyFrame* pKFi : vpNonFixedKFs)
    {
        if(pKFi->isBad())
            continue;

        const int nIDi = pKFi->mnId;

        if(sIdKF.count(nIDi)) // It has already added in the corrected merge KFs
            continue;

        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        Sophus::SE3d Tcw = pKFi->GetPose().cast<double>();
        g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);

        vScw[nIDi] = Siw;
        VSim3->setEstimate(Siw);

        VSim3->setFixed(false);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;

        sIdKF.insert(nIDi);

        vpGoodPose[nIDi] = false;
        vpBadPose[nIDi] = true;
    }

    vector<KeyFrame*> vpKFs;
    vpKFs.reserve(vpFixedKFs.size() + vpFixedCorrectedKFs.size() + vpNonFixedKFs.size());
    vpKFs.insert(vpKFs.end(),vpFixedKFs.begin(),vpFixedKFs.end());
    vpKFs.insert(vpKFs.end(),vpFixedCorrectedKFs.begin(),vpFixedCorrectedKFs.end());
    vpKFs.insert(vpKFs.end(),vpNonFixedKFs.begin(),vpNonFixedKFs.end());
    set<KeyFrame*> spKFs(vpKFs.begin(), vpKFs.end());

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    for(KeyFrame* pKFi : vpKFs)
    {
        int num_connections = 0;
        const int nIDi = pKFi->mnId;

        g2o::Sim3 correctedSwi;
        g2o::Sim3 Swi;

        if(vpGoodPose[nIDi])
            correctedSwi = vCorrectedSwc[nIDi];
        if(vpBadPose[nIDi])
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKFi = pKFi->GetParent();

        // Spanning tree edge
        if(pParentKFi && spKFs.find(pParentKFi) != spKFs.end())
        {
            int nIDj = pParentKFi->mnId;

            g2o::Sim3 Sjw;
            bool bHasRelation = false;

            if(vpGoodPose[nIDi] && vpGoodPose[nIDj])
            {
                Sjw = vCorrectedSwc[nIDj].inverse();
                bHasRelation = true;
            }
            else if(vpBadPose[nIDi] && vpBadPose[nIDj])
            {
                Sjw = vScw[nIDj];
                bHasRelation = true;
            }

            if(bHasRelation)
            {
                g2o::Sim3 Sji = Sjw * Swi;

                g2o::EdgeSim3* e = new g2o::EdgeSim3();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                e->setMeasurement(Sji);

                e->information() = matLambda;
                optimizer.addEdge(e);
                num_connections++;
            }

        }

        // Loop edges
        const set<KeyFrame*> sLoopEdges = pKFi->GetLoopEdges();
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(spKFs.find(pLKF) != spKFs.end() && pLKF->mnId<pKFi->mnId)
            {
                g2o::Sim3 Slw;
                bool bHasRelation = false;

                if(vpGoodPose[nIDi] && vpGoodPose[pLKF->mnId])
                {
                    Slw = vCorrectedSwc[pLKF->mnId].inverse();
                    bHasRelation = true;
                }
                else if(vpBadPose[nIDi] && vpBadPose[pLKF->mnId])
                {
                    Slw = vScw[pLKF->mnId];
                    bHasRelation = true;
                }


                if(bHasRelation)
                {
                    g2o::Sim3 Sli = Slw * Swi;
                    g2o::EdgeSim3* el = new g2o::EdgeSim3();
                    el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                    el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    el->setMeasurement(Sli);
                    el->information() = matLambda;
                    optimizer.addEdge(el);
                    num_connections++;
                }
            }
        }

        // Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKFi->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKFi && !pKFi->hasChild(pKFn) && !sLoopEdges.count(pKFn) && spKFs.find(pKFn) != spKFs.end())
            {
                if(!pKFn->isBad() && pKFn->mnId<pKFi->mnId)
                {

                    g2o::Sim3 Snw =  vScw[pKFn->mnId];
                    bool bHasRelation = false;

                    if(vpGoodPose[nIDi] && vpGoodPose[pKFn->mnId])
                    {
                        Snw = vCorrectedSwc[pKFn->mnId].inverse();
                        bHasRelation = true;
                    }
                    else if(vpBadPose[nIDi] && vpBadPose[pKFn->mnId])
                    {
                        Snw = vScw[pKFn->mnId];
                        bHasRelation = true;
                    }

                    if(bHasRelation)
                    {
                        g2o::Sim3 Sni = Snw * Swi;

                        g2o::EdgeSim3* en = new g2o::EdgeSim3();
                        en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                        en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                        en->setMeasurement(Sni);
                        en->information() = matLambda;
                        optimizer.addEdge(en);
                        num_connections++;
                    }
                }
            }
        }

        if(num_connections == 0 )
        {
            Verbose::PrintMess("Opt_Essential: KF " + to_string(pKFi->mnId) + " has 0 connections", Verbose::VERBOSITY_DEBUG);
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(KeyFrame* pKFi : vpNonFixedKFs)
    {
        if(pKFi->isBad())
            continue;

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        double s = CorrectedSiw.scale();
        Sophus::SE3d Tiw(CorrectedSiw.rotation(),CorrectedSiw.translation() / s);

        pKFi->mTcwBefMerge = pKFi->GetPose();
        pKFi->mTwcBefMerge = pKFi->GetPoseInverse();
        pKFi->SetPose(Tiw.cast<float>());
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(MapPoint* pMPi : vpNonCorrectedMPs)
    {
        if(pMPi->isBad())
            continue;

        KeyFrame* pRefKF = pMPi->GetReferenceKeyFrame();
        while(pRefKF->isBad())
        {
            if(!pRefKF)
            {
                Verbose::PrintMess("MP " + to_string(pMPi->mnId) + " without a valid reference KF", Verbose::VERBOSITY_DEBUG);
                break;
            }

            pMPi->EraseObservation(pRefKF);
            pRefKF = pMPi->GetReferenceKeyFrame();
        }

        if(vpBadPose[pRefKF->mnId])
        {
            Sophus::SE3f TNonCorrectedwr = pRefKF->mTwcBefMerge;
            Sophus::SE3f Twr = pRefKF->GetPoseInverse();

            Eigen::Vector3f eigCorrectedP3Dw = Twr * TNonCorrectedwr.inverse() * pMPi->GetWorldPos();
            pMPi->SetWorldPos(eigCorrectedP3Dw);

            pMPi->UpdateNormalAndDepth();
        }
        else
        {
            cout << "ERROR: MapPoint has a reference KF from another map" << endl;
        }

    }
}

int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2,
                            const bool bFixScale, Eigen::Matrix<double,7,7> &mAcumHessian, const bool bAllPoints)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Camera poses
    const Eigen::Matrix3f R1w = pKF1->GetRotation();
    const Eigen::Vector3f t1w = pKF1->GetTranslation();
    const Eigen::Matrix3f R2w = pKF2->GetRotation();
    const Eigen::Vector3f t2w = pKF2->GetTranslation();

    // Set Sim3 vertex
    ORB_SLAM3::VertexSim3Expmap * vSim3 = new ORB_SLAM3::VertexSim3Expmap();
    vSim3->_fix_scale=bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->pCamera1 = pKF1->mpCamera;
    vSim3->pCamera2 = pKF2->mpCamera;
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<ORB_SLAM3::EdgeSim3ProjectXYZ*> vpEdges12;
    vector<ORB_SLAM3::EdgeInverseSim3ProjectXYZ*> vpEdges21;
    vector<size_t> vnIndexEdge;
    vector<bool> vbIsInKF2;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);
    vbIsInKF2.reserve(2*N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;
    int nBadMPs = 0;
    int nInKF2 = 0;
    int nOutKF2 = 0;
    int nMatchWithoutMP = 0;

    vector<int> vIdsOnlyInKF2;

    for(int i=0; i<N; i++)
    {
        if(!vpMatches1[i])
            continue;

        MapPoint* pMP1 = vpMapPoints1[i];
        MapPoint* pMP2 = vpMatches1[i];

        const int id1 = 2*i+1;
        const int id2 = 2*(i+1);

        const int i2 = get<0>(pMP2->GetIndexInKeyFrame(pKF2));

        Eigen::Vector3f P3D1c;
        Eigen::Vector3f P3D2c;

        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad())
            {
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                Eigen::Vector3f P3D1w = pMP1->GetWorldPos();
                P3D1c = R1w*P3D1w + t1w;
                vPoint1->setEstimate(P3D1c.cast<double>());
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                Eigen::Vector3f P3D2w = pMP2->GetWorldPos();
                P3D2c = R2w*P3D2w + t2w;
                vPoint2->setEstimate(P3D2c.cast<double>());
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
            {
                nBadMPs++;
                continue;
            }
        }
        else
        {
            nMatchWithoutMP++;

            //TODO The 3D position in KF1 doesn't exist
            if(!pMP2->isBad())
            {
                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                Eigen::Vector3f P3D2w = pMP2->GetWorldPos();
                P3D2c = R2w*P3D2w + t2w;
                vPoint2->setEstimate(P3D2c.cast<double>());
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);

                vIdsOnlyInKF2.push_back(id2);
            }
            continue;
        }

        if(i2<0 && !bAllPoints)
        {
            Verbose::PrintMess("    Remove point -> i2: " + to_string(i2) + "; bAllPoints: " + to_string(bAllPoints), Verbose::VERBOSITY_DEBUG);
            continue;
        }

        if(P3D2c(2) < 0)
        {
            Verbose::PrintMess("Sim3: Z coordinate is negative", Verbose::VERBOSITY_DEBUG);
            continue;
        }

        nCorrespondences++;

        // Set edge x1 = S12*X2
        Eigen::Matrix<double,2,1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        ORB_SLAM3::EdgeSim3ProjectXYZ* e12 = new ORB_SLAM3::EdgeSim3ProjectXYZ();

        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double,2,1> obs2;
        cv::KeyPoint kpUn2;
        bool inKF2;
        if(i2 >= 0)
        {
            kpUn2 = pKF2->mvKeysUn[i2];
            obs2 << kpUn2.pt.x, kpUn2.pt.y;
            inKF2 = true;

            nInKF2++;
        }
        else
        {
            float invz = 1/P3D2c(2);
            float x = P3D2c(0)*invz;
            float y = P3D2c(1)*invz;

            obs2 << x, y;
            kpUn2 = cv::KeyPoint(cv::Point2f(x, y), pMP2->mnTrackScaleLevel);

            inKF2 = false;
            nOutKF2++;
        }

        ORB_SLAM3::EdgeInverseSim3ProjectXYZ* e21 = new ORB_SLAM3::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);

        vbIsInKF2.push_back(inKF2);
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    int nBad=0;
    int nBadOutKF2 = 0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        ORB_SLAM3::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        ORB_SLAM3::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<ORB_SLAM3::EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i]=static_cast<ORB_SLAM3::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;

            if(!vbIsInKF2[i])
            {
                nBadOutKF2++;
            }
            continue;
        }

        //Check if remove the robust adjustment improve the result
        e12->setRobustKernel(0);
        e21->setRobustKernel(0);
    }

    int nMoreIterations;
    if(nBad>0)
        nMoreIterations=10;
    else
        nMoreIterations=5;

    if(nCorrespondences-nBad<10)
        return 0;

    // Optimize again only with inliers
    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    mAcumHessian = Eigen::MatrixXd::Zero(7, 7);
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        ORB_SLAM3::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        ORB_SLAM3::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        e12->computeError();
        e21->computeError();

        if(e12->chi2()>th2 || e21->chi2()>th2){
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
        }
        else{
            nIn++;
        }
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12= vSim3_recov->estimate();

    return nIn;
}

void Optimizer::LocalInertialBA(KeyFrame *pKF, bool *pbStopFlag, Map *pMap, int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges, bool bLarge, bool bRecInit)
{
    Map* pCurrentMap = pKF->GetMap();

    int maxOpt=10;
    int opt_it=10;
    if(bLarge)
    {
        maxOpt=25;
        opt_it=4;
    }
    const int Nd = std::min((int)pCurrentMap->KeyFramesInMap()-2,maxOpt);
    const unsigned long maxKFid = pKF->mnId;

    vector<KeyFrame*> vpOptimizableKFs;
    const vector<KeyFrame*> vpNeighsKFs = pKF->GetVectorCovisibleKeyFrames();
    list<KeyFrame*> lpOptVisKFs;

    vpOptimizableKFs.reserve(Nd);
    vpOptimizableKFs.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    for(int i=1; i<Nd; i++)
    {
        if(vpOptimizableKFs.back()->mPrevKF)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pKF->mnId;
        }
        else
            break;
    }

    int N = vpOptimizableKFs.size();

    // Optimizable points seen by temporal optimizable keyframes
    list<MapPoint*> lLocalMapPoints;
    for(int i=0; i<N; i++)
    {
        vector<MapPoint*> vpMPs = vpOptimizableKFs[i]->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
        }
    }

    // Fixed Keyframe: First frame previous KF to optimization window)
    list<KeyFrame*> lFixedKeyFrames;
    if(vpOptimizableKFs.back()->mPrevKF)
    {
        lFixedKeyFrames.push_back(vpOptimizableKFs.back()->mPrevKF);
        vpOptimizableKFs.back()->mPrevKF->mnBAFixedForKF=pKF->mnId;
    }
    else
    {
        vpOptimizableKFs.back()->mnBALocalForKF=0;
        vpOptimizableKFs.back()->mnBAFixedForKF=pKF->mnId;
        lFixedKeyFrames.push_back(vpOptimizableKFs.back());
        vpOptimizableKFs.pop_back();
    }

    // Optimizable visual KFs
    const int maxCovKF = 0;
    for(int i=0, iend=vpNeighsKFs.size(); i<iend; i++)
    {
        if(lpOptVisKFs.size() >= maxCovKF)
            break;

        KeyFrame* pKFi = vpNeighsKFs[i];
        if(pKFi->mnBALocalForKF == pKF->mnId || pKFi->mnBAFixedForKF == pKF->mnId)
            continue;
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
        {
            lpOptVisKFs.push_back(pKFi);

            vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
            for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
            {
                MapPoint* pMP = *vit;
                if(pMP)
                    if(!pMP->isBad())
                        if(pMP->mnBALocalForKF!=pKF->mnId)
                        {
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF=pKF->mnId;
                        }
            }
        }
    }

    // Fixed KFs which are not covisible optimizable
    const int maxFixKF = 200;

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,tuple<int,int>> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                {
                    lFixedKeyFrames.push_back(pKFi);
                    break;
                }
            }
        }
        if(lFixedKeyFrames.size()>=maxFixKF)
            break;
    }

    bool bNonFixed = (lFixedKeyFrames.size() == 0);

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    if(bLarge)
    {
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        solver->setUserLambdaInit(1e-2); // to avoid iterating for finding optimal lambda
        optimizer.setAlgorithm(solver);
    }
    else
    {
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        solver->setUserLambdaInit(1e0);
        optimizer.setAlgorithm(solver);
    }


    // Set Local temporal KeyFrame vertices
    N=vpOptimizableKFs.size();
    for(int i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        if(pKFi->bImu)
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->mnId)+1);
            VV->setFixed(false);
            optimizer.addVertex(VV);
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid+3*(pKFi->mnId)+2);
            VG->setFixed(false);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid+3*(pKFi->mnId)+3);
            VA->setFixed(false);
            optimizer.addVertex(VA);
        }
    }

    // Set Local visual KeyFrame vertices
    for(list<KeyFrame*>::iterator it=lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it!=itEnd; it++)
    {
        KeyFrame* pKFi = *it;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);
    }

    // Set Fixed KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        if(pKFi->bImu) // This should be done only for keyframe just before temporal window
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->mnId)+1);
            VV->setFixed(true);
            optimizer.addVertex(VV);
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid+3*(pKFi->mnId)+2);
            VG->setFixed(true);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid+3*(pKFi->mnId)+3);
            VA->setFixed(true);
            optimizer.addVertex(VA);
        }
    }

    // Create intertial constraints
    vector<EdgeInertial*> vei(N,(EdgeInertial*)NULL);
    vector<EdgeGyroRW*> vegr(N,(EdgeGyroRW*)NULL);
    vector<EdgeAccRW*> vear(N,(EdgeAccRW*)NULL);

    for(int i=0;i<N;i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        if(!pKFi->mPrevKF)
        {
            cout << "NOT INERTIAL LINK TO PREVIOUS FRAME!!!!" << endl;
            continue;
        }
        if(pKFi->bImu && pKFi->mPrevKF->bImu && pKFi->mpImuPreintegrated)
        {
            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+1);
            g2o::HyperGraph::Vertex* VG1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+2);
            g2o::HyperGraph::Vertex* VA1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+3);
            g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+1);
            g2o::HyperGraph::Vertex* VG2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+2);
            g2o::HyperGraph::Vertex* VA2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+3);

            if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            {
                cerr << "Error " << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                continue;
            }

            vei[i] = new EdgeInertial(pKFi->mpImuPreintegrated);

            vei[i]->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            vei[i]->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            vei[i]->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
            vei[i]->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
            vei[i]->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            vei[i]->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

            if(i==N-1 || bRecInit)
            {
                // All inertial residuals are included without robust cost function, but not that one linking the
                // last optimizable keyframe inside of the local window and the first fixed keyframe out. The
                // information matrix for this measurement is also downweighted. This is done to avoid accumulating
                // error due to fixing variables.
                g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
                vei[i]->setRobustKernel(rki);
                if(i==N-1)
                    vei[i]->setInformation(vei[i]->information()*1e-2);
                rki->setDelta(sqrt(16.92));
            }
            optimizer.addEdge(vei[i]);

            vegr[i] = new EdgeGyroRW();
            vegr[i]->setVertex(0,VG1);
            vegr[i]->setVertex(1,VG2);
            Eigen::Matrix3d InfoG = pKFi->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
            vegr[i]->setInformation(InfoG);
            optimizer.addEdge(vegr[i]);

            vear[i] = new EdgeAccRW();
            vear[i]->setVertex(0,VA1);
            vear[i]->setVertex(1,VA2);
            Eigen::Matrix3d InfoA = pKFi->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
            vear[i]->setInformation(InfoA);           

            optimizer.addEdge(vear[i]);
        }
        else
            cout << "ERROR building inertial edge" << endl;
    }

    // Set MapPoint vertices
    const int nExpectedSize = (N+lFixedKeyFrames.size())*lLocalMapPoints.size();

    // Mono
    vector<EdgeMono*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    // Stereo
    vector<EdgeStereo*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);



    const float thHuberMono = sqrt(5.991);
    const float chi2Mono2 = 5.991;
    const float thHuberStereo = sqrt(7.815);
    const float chi2Stereo2 = 7.815;

    const unsigned long iniMPid = maxKFid*5;

    map<int,int> mVisEdges;
    for(int i=0;i<N;i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];
        mVisEdges[pKFi->mnId] = 0;
    }
    for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
    {
        mVisEdges[(*lit)->mnId] = 0;
    }

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());

        unsigned long id = pMP->mnId+iniMPid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

        // Create visual constraints
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
                continue;

            if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            {
                const int leftIndex = get<0>(mit->second);

                cv::KeyPoint kpUn;

                // Monocular left observation
                if(leftIndex != -1 && pKFi->mvuRight[leftIndex]<0)
                {
                    mVisEdges[pKFi->mnId]++;

                    kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMono* e = new EdgeMono(0);

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pKFi->mpCamera->uncertainty2(obs);

                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                // Stereo-observation
                else if(leftIndex != -1)// Stereo observation
                {
                    kpUn = pKFi->mvKeysUn[leftIndex];
                    mVisEdges[pKFi->mnId]++;

                    const float kp_ur = pKFi->mvuRight[leftIndex];
                    Eigen::Matrix<double,3,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereo* e = new EdgeStereo(0);

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pKFi->mpCamera->uncertainty2(obs.head(2));

                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }

                // Monocular right observation
                if(pKFi->mpCamera2){
                    int rightIndex = get<1>(mit->second);

                    if(rightIndex != -1 ){
                        rightIndex -= pKFi->NLeft;
                        mVisEdges[pKFi->mnId]++;

                        Eigen::Matrix<double,2,1> obs;
                        cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                        obs << kp.pt.x, kp.pt.y;

                        EdgeMono* e = new EdgeMono(1);

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pKFi->mpCamera->uncertainty2(obs);

                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave]/unc2;
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);
                        vpEdgesMono.push_back(e);
                        vpEdgeKFMono.push_back(pKFi);
                        vpMapPointEdgeMono.push_back(pMP);
                    }
                }
            }
        }
    }

    // ---- MapLine vertices: the points-parity block. Every line observed by
    // the window enters BA as a 4-DoF orthonormal vertex with one plane edge
    // per observing keyframe -- refined by ALL its observations, not just the
    // 2 planes that triangulated it.
    // Line vertex ids live above the point id range (point ids are
    // pMP->mnId + iniMPid + 1 with pMP->mnId < MapPoint::nNextId).
    const unsigned long iniMLid = iniMPid + MapPoint::nNextId + 2;
    // Observations required before BA may treat a line as an UNKNOWN. The
    // vertex is 6 DoF with 2 of them gauge (sliding along the line, which no
    // observation can see), so 4 are observable and each observation gives 2
    // rows -> 2 would just determine it; require 3 for margin. LM damping
    // handles the gauge directions, exactly as PLVS/ORB-LINE-SLAM do.
    const size_t LINE_BA_MIN_OBS = 4;   // 3-obs lines measured unstable (43-137 cm/call)

    list<MapLine*> lLocalMapLines;
    for(int i=0; MapLine::kVoteInPose && i<N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];
        for(MapLine* pML : pKFi->mvpMapLines)
        {
            if(!pML || pML->isBad() || pML->mnValidated < 2) continue;
            if(pML->mnBALocalForKF == pKF->mnId) continue;
            pML->mnBALocalForKF = pKF->mnId;
            lLocalMapLines.push_back(pML);
        }
    }

    vector<EdgeLine*> vpEdgesLine;
    vector<KeyFrame*> vpEdgeKFLine;
    vector<MapLine*> vpMapLineEdge;
    vector<int> vpEdgeIdxLine;          // observation index within that KF
    list<MapLine*> lLinesInGraph;

    const float chi2Line2 = 5.991;
    for(list<MapLine*>::iterator lit=lLocalMapLines.begin(), lend=lLocalMapLines.end(); lit!=lend; lit++)
    {
        MapLine* pML = *lit;
        const map<KeyFrame*,int> obs = pML->GetObservations();

        // Only keyframes already in the graph constrain the line (we do not
        // enlarge the fixed set for lines); a 4-DoF vertex needs >= 2 edges
        // (4 rows) to be determined, else it is left out untouched.
        vector<pair<KeyFrame*,int>> vUsable;
        for(map<KeyFrame*,int>::const_iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId) continue;
            if(pKFi->isBad() || pKFi->GetMap() != pCurrentMap) continue;
            if(mit->second < 0 || mit->second >= (int)pKFi->mvLines.size()) continue;
            vUsable.push_back(make_pair(pKFi, mit->second));
        }
        if(vUsable.size() < 2) continue;

        // A 3D line has 4 DoF. Each observation contributes ~ONE independent
        // row: its two endpoint residuals both test the same interpretation
        // plane, and on a short segment b1 ~ b2 makes them near-identical.
        // So a 2-observation line is UNDERDETERMINED and the optimizer slides
        // it along the null space -- measured at 60-103 cm per BA call, every
        // call, never converging (6+ observation lines move 4-18 cm).
        // Below the determinacy bar the landmark stays FIXED: its edges still
        // constrain the POSE exactly as before, but BA may not invent its
        // geometry. This is what MapPointCulling's 3-observation rule does
        // for points, stated in the form a line needs.
        // Point-supported lines enter FIXED: their geometry is fit to the
        // point cloud, which BA is already refining. Their edges still
        // constrain the poses.
        const bool bDetermined = (vUsable.size() >= LINE_BA_MIN_OBS)
                                 && !pML->mbSupportFitOk;

        Eigen::Vector3f vS, vE; pML->GetEndpoints(vS, vE);
        VertexLine* vLine = new VertexLine(vS, vE);
        vLine->setId(pML->mnId + iniMLid + 1);
        vLine->setFixed(!bDetermined);
        vLine->setMarginalized(bDetermined);
        optimizer.addVertex(vLine);
        if(bDetermined) lLinesInGraph.push_back(pML);

        for(size_t k=0; k<vUsable.size(); k++)
        {
            KeyFrame* pKFi = vUsable[k].first;
            const LineObs &lo = pKFi->mvLines[vUsable[k].second];

            const float f0 = pKFi->mpCamera ? pKFi->mpCamera->getParameter(0) : 400.f;
            EdgeLine* el = new EdgeLine(lo.b1u, lo.b2u, f0, lo.cam);
            el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vLine));
            el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
            el->setMeasurement(Eigen::Vector3d::Zero());
            // residual is in PIXELS now: sigma = 3 px, same numbers as points
            el->setInformation(Eigen::Matrix2d::Identity() / 9.0);
            g2o::RobustKernelHuber* rkl = new g2o::RobustKernelHuber;
            el->setRobustKernel(rkl);
            rkl->setDelta(sqrt(chi2Line2));
            optimizer.addEdge(el);
            vpEdgesLine.push_back(el);
            vpEdgeKFLine.push_back(pKFi);
            vpMapLineEdge.push_back(pML);
            vpEdgeIdxLine.push_back(vUsable[k].second);
        }
    }

    //cout << "Total map points: " << lLocalMapPoints.size() << endl;
    for(map<int,int>::iterator mit=mVisEdges.begin(), mend=mVisEdges.end(); mit!=mend; mit++)
    {
        assert(mit->second>=3);
    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    float err = optimizer.activeRobustChi2();
    optimizer.optimize(opt_it); // Originally to 2
    float err_end = optimizer.activeRobustChi2();
    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

    // Check inlier observations
    // Mono
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        EdgeMono* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];
        bool bClose = pMP->mTrackDepth<10.f;

        if(pMP->isBad())
            continue;

        if((e->chi2()>chi2Mono2 && !bClose) || (e->chi2()>1.5f*chi2Mono2 && bClose) || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }


    // Stereo
    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        EdgeStereo* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>chi2Stereo2)
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Lines: same inlier discipline as points -- chi2 gate plus the
    // mirror-solution (cheirality) guard.
    vector<pair<KeyFrame*,MapLine*> > vToEraseLine;
    vToEraseLine.reserve(vpEdgesLine.size());
    for(size_t i=0, iend=vpEdgesLine.size(); i<iend;i++)
    {
        EdgeLine* el = vpEdgesLine[i];
        MapLine* pML = vpMapLineEdge[i];
        if(pML->isBad())
            continue;
        if(el->chi2()>chi2Line2 || !el->isDepthPositive())
            vToEraseLine.push_back(make_pair(vpEdgeKFLine[i],pML));
    }

    // Get Map Mutex and erase outliers
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);


    // TODO: Some convergence problems have been detected here
    if((2*err < err_end || isnan(err) || isnan(err_end)) && !bLarge) //bGN)
    {
        cout << "FAIL LOCAL-INERTIAL BA!!!!" << endl;
        return;
    }



    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    for(size_t i=0;i<vToEraseLine.size();i++)
    {
        KeyFrame* pKFi = vToEraseLine[i].first;
        MapLine* pMLi = vToEraseLine[i].second;
        pKFi->EraseMapLineMatch(pMLi);
        pMLi->EraseObservation(pKFi);
    }

    std::cout << "[AUDIT] LBA lines: " << lLinesInGraph.size() << " vertices / "
              << vpEdgesLine.size() << " edges in graph, "
              << vToEraseLine.size() << " obs erased as outliers\n";

    for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
        (*lit)->mnBAFixedForKF = 0;

    // Recover optimized data
    // Local temporal Keyframes
    N=vpOptimizableKFs.size();
    for(int i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
        pKFi->SetPose(Tcw);
        pKFi->mnBALocalForKF=0;

        if(pKFi->bImu)
        {
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
            pKFi->SetVelocity(VV->estimate().cast<float>());
            VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
            VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
            Vector6d b;
            b << VG->estimate(), VA->estimate();
            pKFi->SetNewBias(IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]));

        }
    }

    // Local visual KeyFrame
    for(list<KeyFrame*>::iterator it=lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it!=itEnd; it++)
    {
        KeyFrame* pKFi = *it;
        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
        pKFi->SetPose(Tcw);
        pKFi->mnBALocalForKF=0;
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+iniMPid+1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();
    }

    // Point-supported lines ride their points, which BA just refined.
    for(list<MapLine*>::iterator lit=lLocalMapLines.begin(), lend=lLocalMapLines.end(); lit!=lend; lit++)
    {
        MapLine* pML = *lit;
        if(pML && !pML->isBad() && pML->SupportCount() >= 3)
            pML->RefitFromPoints();
    }

    //Lines: write the refined Plucker back (SetPlucker re-normalises d and
    //re-orthogonalises m against d, and slides the extent onto the new line).
    // Shift bucketed by how many observations constrained the line. A 4-DoF
    // line needs >= 4 independent rows; each edge contributes ~1 (its two
    // endpoint rows are nearly the same constraint on a short segment), so
    // 2-observation lines are UNDERDETERMINED and BA slides them freely.
    std::map<MapLine*,int> nEdgeOf;
    for(size_t i2=0;i2<vpMapLineEdge.size();i2++) nEdgeOf[vpMapLineEdge[i2]]++;
    double bMove[4] = {0,0,0,0}; int bN[4] = {0,0,0,0};
    double dMove = 0.0, dRot = 0.0; int nMoved = 0;
    for(list<MapLine*>::iterator lit=lLinesInGraph.begin(), lend=lLinesInGraph.end(); lit!=lend; lit++)
    {
        MapLine* pML = *lit;
        VertexLine* vLine = static_cast<VertexLine*>(optimizer.vertex(pML->mnId + iniMLid + 1));
        if(!vLine) continue;
        const Eigen::Vector3d S = vLine->estimate().head<3>();
        const Eigen::Vector3d E = vLine->estimate().tail<3>();
        if(!S.allFinite() || !E.allFinite()) continue;
        {   // WHAT BA DID TO THIS LANDMARK: before/after endpoints, so the
            // collapse (both endpoints sliding onto each other -- pure gauge,
            // invisible to the residual) can be SEEN rather than inferred.
            static long nba = 0; static std::ofstream bf;
            if(!bf.is_open()){ bf.open("ba_line_motion.csv");
                bf << "ba,id,nobs,sx0,sy0,sz0,ex0,ey0,ez0,sx1,sy1,sz1,ex1,ey1,ez1\n"; }
            Eigen::Vector3f B1,B2; pML->GetEndpoints(B1,B2);
            if(nba < 400000)
                bf << nba << "," << pML->mnId << "," << nEdgeOf[pML] << ","
                   << B1(0)<<","<<B1(1)<<","<<B1(2)<<","
                   << B2(0)<<","<<B2(1)<<","<<B2(2)<<","
                   << S(0)<<","<<S(1)<<","<<S(2)<<","
                   << E(0)<<","<<E(1)<<","<<E(2)<<"\n";
            nba++;
        }
        // ---- GUARD + RE-CLIP -------------------------------------------
        // The 2 DoF of sliding an endpoint ALONG the line are GAUGE: the
        // residual cannot see them, so LM slides both endpoints together and
        // the segment collapses to a point (measured: most landmarks, worse
        // with MORE observations). PLVS blocks this with a 3D endpoint term
        // that needs depth; monocular has no depth, so instead:
        //   1. refuse a collapsed / exploded update,
        //   2. take only the LINE from BA and RE-DERIVE the endpoints by
        //      intersecting the observed bearings with it (union over the
        //      observing keyframes). The endpoints stop being free at all.
        const double lenNew = (E - S).norm();
        if(!(lenNew > 0.02 && lenNew < 10.0)) continue;  // keep the pre-BA extent
        // how far did BA actually move this landmark? midpoint shift +
        // direction change. Zero here means the vertex never budged.
        Eigen::Vector3f S0, E0; pML->GetEndpoints(S0, E0);
        const Eigen::Vector3f d0 = pML->GetDirection();
        // GAUGE-INVARIANT motion: perpendicular distance between the old and
        // new INFINITE lines. Sliding the endpoints along the line is gauge
        // (unobservable) and must not be counted as the landmark moving.
        const Eigen::Vector3d dn_ = (E - S).normalized();
        const Eigen::Vector3d w_  = 0.5*(S+E) - 0.5*(S0+E0).cast<double>();
        const double sh = (w_ - dn_*(w_.dot(dn_))).norm();
        const Eigen::Vector3d dn = (E - S).normalized();
        dMove += sh;
        dRot  += std::acos(std::min(1.0, std::fabs(dn.dot(d0.cast<double>()))));
        nMoved++;
        const int ne = nEdgeOf[pML];
        const int bk = ne <= 2 ? 0 : (ne <= 3 ? 1 : (ne <= 5 ? 2 : 3));
        bMove[bk] += sh; bN[bk]++;
        pML->SetEndpoints(S.cast<float>(), E.cast<float>());
        {   // RE-DERIVE the extent from the observations (kills the gauge)
            const Eigen::Vector3f dW = pML->GetDirection();
            const Eigen::Vector3f p0 = dW.cross(pML->GetMoment());
            float umin = 1e9f, umax = -1e9f; int nUsed = 0;
            for(size_t q = 0; q < vpMapLineEdge.size(); q++)
            {
                if(vpMapLineEdge[q] != pML) continue;
                KeyFrame* pKq = vpEdgeKFLine[q];
                const int iq = vpEdgeIdxLine[q];
                if(!pKq || pKq->isBad() || iq < 0 || iq >= (int)pKq->mvLines.size()) continue;
                const LineObs &lq = pKq->mvLines[iq];
                Sophus::SE3f Tq = pKq->GetPose();
                if(lq.cam == 1 && pKq->mpCamera2) Tq = pKq->GetRelativePoseTrl() * Tq;
                const Eigen::Matrix3f Rq = Tq.rotationMatrix();
                const Eigen::Vector3f tq = Tq.translation();
                const Eigen::Vector3f dC = Rq * dW;
                const Eigen::Vector3f mC = Rq * pML->GetMoment() + tq.cross(dC);
                for(const Eigen::Vector3f& b : {lq.b1u, lq.b2u}){
                    const Eigen::Vector3f cr = b.cross(dC);
                    const float den = cr.squaredNorm();
                    if(den < 1e-10f) continue;
                    const float sdep = mC.dot(cr) / den;
                    if(!(sdep > 0.05f && sdep < 40.f)) continue;
                    const Eigen::Vector3f xw = Rq.transpose() * (sdep * b - tq);
                    const float u = (xw - p0).dot(dW);
                    umin = std::min(umin, u); umax = std::max(umax, u);
                    nUsed++;
                }
            }
            if(nUsed >= 2 && umax - umin > 0.02f && umax - umin < 20.f)
                pML->SetEndpoints(p0 + dW * umin, p0 + dW * umax);
        }
    }
    if(nMoved){
        std::cout << "[AUDIT] LBA line motion: " << nMoved << " lines, mean shift "
                  << 100.0*dMove/nMoved << " cm, mean rotation "
                  << 180.0/M_PI*dRot/nMoved << " deg | by #obs: ";
        const char* lbl[4] = {"2obs","3obs","4-5obs","6+obs"};
        for(int k2=0;k2<4;k2++)
            std::cout << lbl[k2] << " n=" << bN[k2] << " "
                      << (bN[k2] ? 100.0*bMove[k2]/bN[k2] : 0.0) << "cm  ";
        std::cout << std::endl;
    }

    pMap->IncreaseChangeIndex();
}

Eigen::MatrixXd Optimizer::Marginalize(const Eigen::MatrixXd &H, const int &start, const int &end)
{
    // Goal
    // a  | ab | ac       a*  | 0 | ac*
    // ba | b  | bc  -->  0   | 0 | 0
    // ca | cb | c        ca* | 0 | c*

    // Size of block before block to marginalize
    const int a = start;
    // Size of block to marginalize
    const int b = end-start+1;
    // Size of block after block to marginalize
    const int c = H.cols() - (end+1);

    // Reorder as follows:
    // a  | ab | ac       a  | ac | ab
    // ba | b  | bc  -->  ca | c  | cb
    // ca | cb | c        ba | bc | b

    Eigen::MatrixXd Hn = Eigen::MatrixXd::Zero(H.rows(),H.cols());
    if(a>0)
    {
        Hn.block(0,0,a,a) = H.block(0,0,a,a);
        Hn.block(0,a+c,a,b) = H.block(0,a,a,b);
        Hn.block(a+c,0,b,a) = H.block(a,0,b,a);
    }
    if(a>0 && c>0)
    {
        Hn.block(0,a,a,c) = H.block(0,a+b,a,c);
        Hn.block(a,0,c,a) = H.block(a+b,0,c,a);
    }
    if(c>0)
    {
        Hn.block(a,a,c,c) = H.block(a+b,a+b,c,c);
        Hn.block(a,a+c,c,b) = H.block(a+b,a,c,b);
        Hn.block(a+c,a,b,c) = H.block(a,a+b,b,c);
    }
    Hn.block(a+c,a+c,b,b) = H.block(a,a,b,b);

    // Perform marginalization (Schur complement)
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Hn.block(a+c,a+c,b,b),Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType singularValues_inv=svd.singularValues();
    for (int i=0; i<b; ++i)
    {
        if (singularValues_inv(i)>1e-6)
            singularValues_inv(i)=1.0/singularValues_inv(i);
        else singularValues_inv(i)=0;
    }
    Eigen::MatrixXd invHb = svd.matrixV()*singularValues_inv.asDiagonal()*svd.matrixU().transpose();
    Hn.block(0,0,a+c,a+c) = Hn.block(0,0,a+c,a+c) - Hn.block(0,a+c,a+c,b)*invHb*Hn.block(a+c,0,b,a+c);
    Hn.block(a+c,a+c,b,b) = Eigen::MatrixXd::Zero(b,b);
    Hn.block(0,a+c,a+c,b) = Eigen::MatrixXd::Zero(a+c,b);
    Hn.block(a+c,0,b,a+c) = Eigen::MatrixXd::Zero(b,a+c);

    // Inverse reorder
    // a*  | ac* | 0       a*  | 0 | ac*
    // ca* | c*  | 0  -->  0   | 0 | 0
    // 0   | 0   | 0       ca* | 0 | c*
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(H.rows(),H.cols());
    if(a>0)
    {
        res.block(0,0,a,a) = Hn.block(0,0,a,a);
        res.block(0,a,a,b) = Hn.block(0,a+c,a,b);
        res.block(a,0,b,a) = Hn.block(a+c,0,b,a);
    }
    if(a>0 && c>0)
    {
        res.block(0,a+b,a,c) = Hn.block(0,a,a,c);
        res.block(a+b,0,c,a) = Hn.block(a,0,c,a);
    }
    if(c>0)
    {
        res.block(a+b,a+b,c,c) = Hn.block(a,a,c,c);
        res.block(a+b,a,c,b) = Hn.block(a,a+c,c,b);
        res.block(a,a+b,b,c) = Hn.block(a+c,a,b,c);
    }

    res.block(a,a,b,b) = Hn.block(a+c,a+c,b,b);

    return res;
}

void Optimizer::InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg, Eigen::Vector3d &ba, bool bMono, Eigen::MatrixXd  &covInertial, bool bFixedVel, bool bGauss, float priorG, float priorA)
{
    Verbose::PrintMess("inertial optimization", Verbose::VERBOSITY_NORMAL);
    int its = 200;
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    if (priorG!=0.f)
        solver->setUserLambdaInit(1e3);

    optimizer.setAlgorithm(solver);

    // Set KeyFrame vertices (fixed poses and optimizable velocities)
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        VertexVelocity* VV = new VertexVelocity(pKFi);
        VV->setId(maxKFid+(pKFi->mnId)+1);
        if (bFixedVel)
            VV->setFixed(true);
        else
            VV->setFixed(false);

        optimizer.addVertex(VV);
    }

    // Biases
    VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
    VG->setId(maxKFid*2+2);
    if (bFixedVel)
        VG->setFixed(true);
    else
        VG->setFixed(false);
    optimizer.addVertex(VG);
    VertexAccBias* VA = new VertexAccBias(vpKFs.front());
    VA->setId(maxKFid*2+3);
    if (bFixedVel)
        VA->setFixed(true);
    else
        VA->setFixed(false);

    optimizer.addVertex(VA);
    // prior acc bias
    Eigen::Vector3f bprior;
    bprior.setZero();

    EdgePriorAcc* epa = new EdgePriorAcc(bprior);
    epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
    double infoPriorA = priorA;
    epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
    optimizer.addEdge(epa);
    EdgePriorGyro* epg = new EdgePriorGyro(bprior);
    epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
    double infoPriorG = priorG;
    epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
    optimizer.addEdge(epg);

    // Gravity and scale
    VertexGDir* VGDir = new VertexGDir(Rwg);
    VGDir->setId(maxKFid*2+4);
    VGDir->setFixed(false);
    optimizer.addVertex(VGDir);
    VertexScale* VS = new VertexScale(scale);
    VS->setId(maxKFid*2+5);
    VS->setFixed(!bMono); // Fixed for stereo case
    optimizer.addVertex(VS);

    // Graph edges
    // IMU links with gravity and scale
    vector<EdgeInertialGS*> vpei;
    vpei.reserve(vpKFs.size());
    vector<pair<KeyFrame*,KeyFrame*> > vppUsedKF;
    vppUsedKF.reserve(vpKFs.size());
    //std::cout << "build optimization graph" << std::endl;

    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        if(pKFi->mPrevKF && pKFi->mnId<=maxKFid)
        {
            if(pKFi->isBad() || pKFi->mPrevKF->mnId>maxKFid)
                continue;
            if(!pKFi->mpImuPreintegrated)
                std::cout << "Not preintegrated measurement" << std::endl;

            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+(pKFi->mPrevKF->mnId)+1);
            g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+(pKFi->mnId)+1);
            g2o::HyperGraph::Vertex* VG = optimizer.vertex(maxKFid*2+2);
            g2o::HyperGraph::Vertex* VA = optimizer.vertex(maxKFid*2+3);
            g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(maxKFid*2+4);
            g2o::HyperGraph::Vertex* VS = optimizer.vertex(maxKFid*2+5);
            if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG << ", "<< VA << ", " << VP2 << ", " << VV2 <<  ", "<< VGDir << ", "<< VS <<endl;

                continue;
            }
            EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
            ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
            ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            ei->setVertex(6,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
            ei->setVertex(7,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));

            vpei.push_back(ei);

            vppUsedKF.push_back(make_pair(pKFi->mPrevKF,pKFi));
            optimizer.addEdge(ei);

        }
    }

    // Compute error for different scales
    std::set<g2o::HyperGraph::Edge*> setEdges = optimizer.edges();

    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(its);

    scale = VS->estimate();

    // Recover optimized data
    // Biases
    VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid*2+2));
    VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid*2+3));
    Vector6d vb;
    vb << VG->estimate(), VA->estimate();
    bg << VG->estimate();
    ba << VA->estimate();
    scale = VS->estimate();


    IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);
    Rwg = VGDir->estimate().Rwg;

    //Keyframes velocities and biases
    const int N = vpKFs.size();
    for(size_t i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;

        VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+(pKFi->mnId)+1));
        Eigen::Vector3d Vw = VV->estimate(); // Velocity is scaled after
        pKFi->SetVelocity(Vw.cast<float>());

        if ((pKFi->GetGyroBias() - bg.cast<float>()).norm() > 0.01)
        {
            pKFi->SetNewBias(b);
            if (pKFi->mpImuPreintegrated)
                pKFi->mpImuPreintegrated->Reintegrate();
        }
        else
            pKFi->SetNewBias(b);


    }
}


void Optimizer::InertialOptimization(Map *pMap, Eigen::Vector3d &bg, Eigen::Vector3d &ba, float priorG, float priorA)
{
    int its = 200; // Check number of iterations
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setUserLambdaInit(1e3);

    optimizer.setAlgorithm(solver);

    // Set KeyFrame vertices (fixed poses and optimizable velocities)
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        VertexVelocity* VV = new VertexVelocity(pKFi);
        VV->setId(maxKFid+(pKFi->mnId)+1);
        VV->setFixed(false);

        optimizer.addVertex(VV);
    }

    // Biases
    VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
    VG->setId(maxKFid*2+2);
    VG->setFixed(false);
    optimizer.addVertex(VG);

    VertexAccBias* VA = new VertexAccBias(vpKFs.front());
    VA->setId(maxKFid*2+3);
    VA->setFixed(false);

    optimizer.addVertex(VA);
    // prior acc bias
    Eigen::Vector3f bprior;
    bprior.setZero();

    EdgePriorAcc* epa = new EdgePriorAcc(bprior);
    epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
    double infoPriorA = priorA;
    epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
    optimizer.addEdge(epa);
    EdgePriorGyro* epg = new EdgePriorGyro(bprior);
    epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
    double infoPriorG = priorG;
    epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
    optimizer.addEdge(epg);

    // Gravity and scale
    VertexGDir* VGDir = new VertexGDir(Eigen::Matrix3d::Identity());
    VGDir->setId(maxKFid*2+4);
    VGDir->setFixed(true);
    optimizer.addVertex(VGDir);
    VertexScale* VS = new VertexScale(1.0);
    VS->setId(maxKFid*2+5);
    VS->setFixed(true); // Fixed since scale is obtained from already well initialized map
    optimizer.addVertex(VS);

    // Graph edges
    // IMU links with gravity and scale
    vector<EdgeInertialGS*> vpei;
    vpei.reserve(vpKFs.size());
    vector<pair<KeyFrame*,KeyFrame*> > vppUsedKF;
    vppUsedKF.reserve(vpKFs.size());

    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        if(pKFi->mPrevKF && pKFi->mnId<=maxKFid)
        {
            if(pKFi->isBad() || pKFi->mPrevKF->mnId>maxKFid)
                continue;

            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+(pKFi->mPrevKF->mnId)+1);
            g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+(pKFi->mnId)+1);
            g2o::HyperGraph::Vertex* VG = optimizer.vertex(maxKFid*2+2);
            g2o::HyperGraph::Vertex* VA = optimizer.vertex(maxKFid*2+3);
            g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(maxKFid*2+4);
            g2o::HyperGraph::Vertex* VS = optimizer.vertex(maxKFid*2+5);
            if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG << ", "<< VA << ", " << VP2 << ", " << VV2 <<  ", "<< VGDir << ", "<< VS <<endl;

                continue;
            }
            EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
            ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
            ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            ei->setVertex(6,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
            ei->setVertex(7,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));

            vpei.push_back(ei);

            vppUsedKF.push_back(make_pair(pKFi->mPrevKF,pKFi));
            optimizer.addEdge(ei);

        }
    }

    // Compute error for different scales
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(its);


    // Recover optimized data
    // Biases
    VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid*2+2));
    VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid*2+3));
    Vector6d vb;
    vb << VG->estimate(), VA->estimate();
    bg << VG->estimate();
    ba << VA->estimate();

    IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);

    //Keyframes velocities and biases
    const int N = vpKFs.size();
    for(size_t i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;

        VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+(pKFi->mnId)+1));
        Eigen::Vector3d Vw = VV->estimate();
        pKFi->SetVelocity(Vw.cast<float>());

        if ((pKFi->GetGyroBias() - bg.cast<float>()).norm() > 0.01)
        {
            pKFi->SetNewBias(b);
            if (pKFi->mpImuPreintegrated)
                pKFi->mpImuPreintegrated->Reintegrate();
        }
        else
            pKFi->SetNewBias(b);
    }
}

void Optimizer::InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale)
{
    int its = 10;
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Set KeyFrame vertices (all variables are fixed)
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        VertexVelocity* VV = new VertexVelocity(pKFi);
        VV->setId(maxKFid+1+(pKFi->mnId));
        VV->setFixed(true);
        optimizer.addVertex(VV);

        // Vertex of fixed biases
        VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
        VG->setId(2*(maxKFid+1)+(pKFi->mnId));
        VG->setFixed(true);
        optimizer.addVertex(VG);
        VertexAccBias* VA = new VertexAccBias(vpKFs.front());
        VA->setId(3*(maxKFid+1)+(pKFi->mnId));
        VA->setFixed(true);
        optimizer.addVertex(VA);
    }

    // Gravity and scale
    VertexGDir* VGDir = new VertexGDir(Rwg);
    VGDir->setId(4*(maxKFid+1));
    VGDir->setFixed(false);
    optimizer.addVertex(VGDir);
    VertexScale* VS = new VertexScale(scale);
    VS->setId(4*(maxKFid+1)+1);
    VS->setFixed(false);
    optimizer.addVertex(VS);

    // Graph edges
    int count_edges = 0;
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        if(pKFi->mPrevKF && pKFi->mnId<=maxKFid)
        {
            if(pKFi->isBad() || pKFi->mPrevKF->mnId>maxKFid)
                continue;
                
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex((maxKFid+1)+pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex((maxKFid+1)+pKFi->mnId);
            g2o::HyperGraph::Vertex* VG = optimizer.vertex(2*(maxKFid+1)+pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VA = optimizer.vertex(3*(maxKFid+1)+pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(4*(maxKFid+1));
            g2o::HyperGraph::Vertex* VS = optimizer.vertex(4*(maxKFid+1)+1);
            if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                Verbose::PrintMess("Error" + to_string(VP1->id()) + ", " + to_string(VV1->id()) + ", " + to_string(VG->id()) + ", " + to_string(VA->id()) + ", " + to_string(VP2->id()) + ", " + to_string(VV2->id()) +  ", " + to_string(VGDir->id()) + ", " + to_string(VS->id()), Verbose::VERBOSITY_NORMAL);

                continue;
            }
            count_edges++;
            EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
            ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
            ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            ei->setVertex(6,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
            ei->setVertex(7,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            ei->setRobustKernel(rk);
            rk->setDelta(1.f);
            optimizer.addEdge(ei);
        }
    }

    // Compute error for different scales
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    float err = optimizer.activeRobustChi2();
    optimizer.optimize(its);
    optimizer.computeActiveErrors();
    float err_end = optimizer.activeRobustChi2();
    // Recover optimized data
    scale = VS->estimate();
    Rwg = VGDir->estimate().Rwg;
}

void Optimizer::LocalBundleAdjustment(KeyFrame* pMainKF,vector<KeyFrame*> vpAdjustKF, vector<KeyFrame*> vpFixedKF, bool *pbStopFlag)
{
    bool bShowImages = false;

    vector<MapPoint*> vpMPs;

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    optimizer.setVerbose(false);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;
    set<KeyFrame*> spKeyFrameBA;

    Map* pCurrentMap = pMainKF->GetMap();

    // Set fixed KeyFrame vertices
    int numInsertedPoints = 0;
    for(KeyFrame* pKFi : vpFixedKF)
    {
        if(pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
        {
            Verbose::PrintMess("ERROR LBA: KF is bad or is not in the current map", Verbose::VERBOSITY_NORMAL);
            continue;
        }

        pKFi->mnBALocalForMerge = pMainKF->mnId;

        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;

        set<MapPoint*> spViewMPs = pKFi->GetMapPoints();
        for(MapPoint* pMPi : spViewMPs)
        {
            if(pMPi)
                if(!pMPi->isBad() && pMPi->GetMap() == pCurrentMap)

                    if(pMPi->mnBALocalForMerge!=pMainKF->mnId)
                    {
                        vpMPs.push_back(pMPi);
                        pMPi->mnBALocalForMerge=pMainKF->mnId;
                        numInsertedPoints++;
                    }
        }

        spKeyFrameBA.insert(pKFi);
    }

    // Set non fixed Keyframe vertices
    set<KeyFrame*> spAdjustKF(vpAdjustKF.begin(), vpAdjustKF.end());
    numInsertedPoints = 0;
    for(KeyFrame* pKFi : vpAdjustKF)
    {
        if(pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
            continue;

        pKFi->mnBALocalForMerge = pMainKF->mnId;

        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;

        set<MapPoint*> spViewMPs = pKFi->GetMapPoints();
        for(MapPoint* pMPi : spViewMPs)
        {
            if(pMPi)
            {
                if(!pMPi->isBad() && pMPi->GetMap() == pCurrentMap)
                {
                    if(pMPi->mnBALocalForMerge != pMainKF->mnId)
                    {
                        vpMPs.push_back(pMPi);
                        pMPi->mnBALocalForMerge = pMainKF->mnId;
                        numInsertedPoints++;
                    }
                }
            }
        }

        spKeyFrameBA.insert(pKFi);
    }

    const int nExpectedSize = (vpAdjustKF.size()+vpFixedKF.size())*vpMPs.size();

    vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    map<KeyFrame*, int> mpObsKFs;
    map<KeyFrame*, int> mpObsFinalKFs;
    map<MapPoint*, int> mpObsMPs;
    for(unsigned int i=0; i < vpMPs.size(); ++i)
    {
        MapPoint* pMPi = vpMPs[i];
        if(pMPi->isBad())
            continue;

        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMPi->GetWorldPos().cast<double>());
        const int id = pMPi->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);


        const map<KeyFrame*,tuple<int,int>> observations = pMPi->GetObservations();
        int nEdges = 0;
        //SET EDGES
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {
            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid || pKF->mnBALocalForMerge != pMainKF->mnId || !pKF->GetMapPoint(get<0>(mit->second)))
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[get<0>(mit->second)];

            if(pKF->mvuRight[get<0>(mit->second)]<0) //Monocular
            {
                mpObsMPs[pMPi]++;
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuber2D);

                e->pCamera = pKF->mpCamera;

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKF);
                vpMapPointEdgeMono.push_back(pMPi);

                mpObsKFs[pKF]++;
            }
            else // RGBD or Stereo
            {
                mpObsMPs[pMPi]+=2;
                Eigen::Matrix<double,3,1> obs;
                const float kp_ur = pKF->mvuRight[get<0>(mit->second)];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuber3D);

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vpEdgeKFStereo.push_back(pKF);
                vpMapPointEdgeStereo.push_back(pMPi);

                mpObsKFs[pKF]++;
            }
        }
    }

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore= true;

    if(pbStopFlag)
        if(*pbStopFlag)
            bDoMore = false;

    map<unsigned long int, int> mWrongObsKF;
    if(bDoMore)
    {
        // Check inlier observations
        int badMonoMP = 0, badStereoMP = 0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                e->setLevel(1);
                badMonoMP++;
            }
            e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            MapPoint* pMP = vpMapPointEdgeStereo[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>7.815 || !e->isDepthPositive())
            {
                e->setLevel(1);
                badStereoMP++;
            }

            e->setRobustKernel(0);
        }
        Verbose::PrintMess("[BA]: First optimization(Huber), there are " + to_string(badMonoMP) + " monocular and " + to_string(badStereoMP) + " stereo bad edges", Verbose::VERBOSITY_DEBUG);

    optimizer.initializeOptimization(0);
    optimizer.optimize(10);
    }

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());
    set<MapPoint*> spErasedMPs;
    set<KeyFrame*> spErasedKFs;

    // Check inlier observations
    int badMonoMP = 0, badStereoMP = 0;
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
            mWrongObsKF[pKFi->mnId]++;
            badMonoMP++;

            spErasedMPs.insert(pMP);
            spErasedKFs.insert(pKFi);
        }
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
            mWrongObsKF[pKFi->mnId]++;
            badStereoMP++;

            spErasedMPs.insert(pMP);
            spErasedKFs.insert(pKFi);
        }
    }

    Verbose::PrintMess("[BA]: Second optimization, there are " + to_string(badMonoMP) + " monocular and " + to_string(badStereoMP) + " sterero bad edges", Verbose::VERBOSITY_DEBUG);

    // Get Map Mutex
    unique_lock<mutex> lock(pMainKF->GetMap()->mMutexMapUpdate);

    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }
    for(unsigned int i=0; i < vpMPs.size(); ++i)
    {
        MapPoint* pMPi = vpMPs[i];
        if(pMPi->isBad())
            continue;

        const map<KeyFrame*,tuple<int,int>> observations = pMPi->GetObservations();
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {
            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid || pKF->mnBALocalForKF != pMainKF->mnId || !pKF->GetMapPoint(get<0>(mit->second)))
                continue;

            if(pKF->mvuRight[get<0>(mit->second)]<0) //Monocular
            {
                mpObsFinalKFs[pKF]++;
            }
            else // RGBD or Stereo
            {
                mpObsFinalKFs[pKF]++;
            }
        }
    }

    // Recover optimized data
    // Keyframes
    for(KeyFrame* pKFi : vpAdjustKF)
    {
        if(pKFi->isBad())
            continue;

        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());

        int numMonoBadPoints = 0, numMonoOptPoints = 0;
        int numStereoBadPoints = 0, numStereoOptPoints = 0;
        vector<MapPoint*> vpMonoMPsOpt, vpStereoMPsOpt;
        vector<MapPoint*> vpMonoMPsBad, vpStereoMPsBad;

        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];
            KeyFrame* pKFedge = vpEdgeKFMono[i];

            if(pKFi != pKFedge)
            {
                continue;
            }

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                numMonoBadPoints++;
                vpMonoMPsBad.push_back(pMP);

            }
            else
            {
                numMonoOptPoints++;
                vpMonoMPsOpt.push_back(pMP);
            }

        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            MapPoint* pMP = vpMapPointEdgeStereo[i];
            KeyFrame* pKFedge = vpEdgeKFMono[i];

            if(pKFi != pKFedge)
            {
                continue;
            }

            if(pMP->isBad())
                continue;

            if(e->chi2()>7.815 || !e->isDepthPositive())
            {
                numStereoBadPoints++;
                vpStereoMPsBad.push_back(pMP);
            }
            else
            {
                numStereoOptPoints++;
                vpStereoMPsOpt.push_back(pMP);
            }
        }

        pKFi->SetPose(Tiw);
    }

    //Points
    for(MapPoint* pMPi : vpMPs)
    {
        if(pMPi->isBad())
            continue;

        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMPi->mnId+maxKFid+1));
        pMPi->SetWorldPos(vPoint->estimate().cast<float>());
        pMPi->UpdateNormalAndDepth();

    }
}


void Optimizer::MergeInertialBA(KeyFrame* pCurrKF, KeyFrame* pMergeKF, bool *pbStopFlag, Map *pMap, LoopClosing::KeyFrameAndPose &corrPoses)
{
    const int Nd = 6;
    const unsigned long maxKFid = pCurrKF->mnId;

    vector<KeyFrame*> vpOptimizableKFs;
    vpOptimizableKFs.reserve(2*Nd);

    // For cov KFS, inertial parameters are not optimized
    const int maxCovKF = 30;
    vector<KeyFrame*> vpOptimizableCovKFs;
    vpOptimizableCovKFs.reserve(maxCovKF);

    // Add sliding window for current KF
    vpOptimizableKFs.push_back(pCurrKF);
    pCurrKF->mnBALocalForKF = pCurrKF->mnId;
    for(int i=1; i<Nd; i++)
    {
        if(vpOptimizableKFs.back()->mPrevKF)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
        }
        else
            break;
    }

    list<KeyFrame*> lFixedKeyFrames;
    if(vpOptimizableKFs.back()->mPrevKF)
    {
        vpOptimizableCovKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
        vpOptimizableKFs.back()->mPrevKF->mnBALocalForKF=pCurrKF->mnId;
    }
    else
    {
        vpOptimizableCovKFs.push_back(vpOptimizableKFs.back());
        vpOptimizableKFs.pop_back();
    }

    // Add temporal neighbours to merge KF (previous and next KFs)
    vpOptimizableKFs.push_back(pMergeKF);
    pMergeKF->mnBALocalForKF = pCurrKF->mnId;

    // Previous KFs
    for(int i=1; i<(Nd/2); i++)
    {
        if(vpOptimizableKFs.back()->mPrevKF)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
        }
        else
            break;
    }

    // We fix just once the old map
    if(vpOptimizableKFs.back()->mPrevKF)
    {
        lFixedKeyFrames.push_back(vpOptimizableKFs.back()->mPrevKF);
        vpOptimizableKFs.back()->mPrevKF->mnBAFixedForKF=pCurrKF->mnId;
    }
    else
    {
        vpOptimizableKFs.back()->mnBALocalForKF=0;
        vpOptimizableKFs.back()->mnBAFixedForKF=pCurrKF->mnId;
        lFixedKeyFrames.push_back(vpOptimizableKFs.back());
        vpOptimizableKFs.pop_back();
    }

    // Next KFs
    if(pMergeKF->mNextKF)
    {
        vpOptimizableKFs.push_back(pMergeKF->mNextKF);
        vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
    }

    while(vpOptimizableKFs.size()<(2*Nd))
    {
        if(vpOptimizableKFs.back()->mNextKF)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mNextKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
        }
        else
            break;
    }

    int N = vpOptimizableKFs.size();

    // Optimizable points seen by optimizable keyframes
    list<MapPoint*> lLocalMapPoints;
    map<MapPoint*,int> mLocalObs;
    for(int i=0; i<N; i++)
    {
        vector<MapPoint*> vpMPs = vpOptimizableKFs[i]->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            // Using mnBALocalForKF we avoid redundance here, one MP can not be added several times to lLocalMapPoints
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pCurrKF->mnId)
                    {
                        mLocalObs[pMP]=1;
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pCurrKF->mnId;
                    }
                    else {
                        mLocalObs[pMP]++;
                    }
        }
    }

    std::vector<std::pair<MapPoint*, int>> pairs;
    pairs.reserve(mLocalObs.size());
    for (auto itr = mLocalObs.begin(); itr != mLocalObs.end(); ++itr)
        pairs.push_back(*itr);
    sort(pairs.begin(), pairs.end(),sortByVal);

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    int i=0;
    for(vector<pair<MapPoint*,int>>::iterator lit=pairs.begin(), lend=pairs.end(); lit!=lend; lit++, i++)
    {
        map<KeyFrame*,tuple<int,int>> observations = lit->first->GetObservations();
        if(i>=maxCovKF)
            break;
        for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pCurrKF->mnId && pKFi->mnBAFixedForKF!=pCurrKF->mnId) // If optimizable or already included...
            {
                pKFi->mnBALocalForKF=pCurrKF->mnId;
                if(!pKFi->isBad())
                {
                    vpOptimizableCovKFs.push_back(pKFi);
                    break;
                }
            }
        }
    }

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e3);

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    // Set Local KeyFrame vertices
    N=vpOptimizableKFs.size();
    for(int i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        if(pKFi->bImu)
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->mnId)+1);
            VV->setFixed(false);
            optimizer.addVertex(VV);
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid+3*(pKFi->mnId)+2);
            VG->setFixed(false);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid+3*(pKFi->mnId)+3);
            VA->setFixed(false);
            optimizer.addVertex(VA);
        }
    }

    // Set Local cov keyframes vertices
    int Ncov=vpOptimizableCovKFs.size();
    for(int i=0; i<Ncov; i++)
    {
        KeyFrame* pKFi = vpOptimizableCovKFs[i];

        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        if(pKFi->bImu)
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->mnId)+1);
            VV->setFixed(false);
            optimizer.addVertex(VV);
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid+3*(pKFi->mnId)+2);
            VG->setFixed(false);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid+3*(pKFi->mnId)+3);
            VA->setFixed(false);
            optimizer.addVertex(VA);
        }
    }

    // Set Fixed KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        if(pKFi->bImu)
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->mnId)+1);
            VV->setFixed(true);
            optimizer.addVertex(VV);
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid+3*(pKFi->mnId)+2);
            VG->setFixed(true);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid+3*(pKFi->mnId)+3);
            VA->setFixed(true);
            optimizer.addVertex(VA);
        }
    }

    // Create intertial constraints
    vector<EdgeInertial*> vei(N,(EdgeInertial*)NULL);
    vector<EdgeGyroRW*> vegr(N,(EdgeGyroRW*)NULL);
    vector<EdgeAccRW*> vear(N,(EdgeAccRW*)NULL);
    for(int i=0;i<N;i++)
    {
        //cout << "inserting inertial edge " << i << endl;
        KeyFrame* pKFi = vpOptimizableKFs[i];

        if(!pKFi->mPrevKF)
        {
            Verbose::PrintMess("NOT INERTIAL LINK TO PREVIOUS FRAME!!!!", Verbose::VERBOSITY_NORMAL);
            continue;
        }
        if(pKFi->bImu && pKFi->mPrevKF->bImu && pKFi->mpImuPreintegrated)
        {
            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+1);
            g2o::HyperGraph::Vertex* VG1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+2);
            g2o::HyperGraph::Vertex* VA1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+3);
            g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+1);
            g2o::HyperGraph::Vertex* VG2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+2);
            g2o::HyperGraph::Vertex* VA2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+3);

            if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            {
                cerr << "Error " << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                continue;
            }

            vei[i] = new EdgeInertial(pKFi->mpImuPreintegrated);

            vei[i]->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            vei[i]->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            vei[i]->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
            vei[i]->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
            vei[i]->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            vei[i]->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

            // TODO Uncomment
            g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
            vei[i]->setRobustKernel(rki);
            rki->setDelta(sqrt(16.92));
            optimizer.addEdge(vei[i]);

            vegr[i] = new EdgeGyroRW();
            vegr[i]->setVertex(0,VG1);
            vegr[i]->setVertex(1,VG2);
            Eigen::Matrix3d InfoG = pKFi->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
            vegr[i]->setInformation(InfoG);
            optimizer.addEdge(vegr[i]);

            vear[i] = new EdgeAccRW();
            vear[i]->setVertex(0,VA1);
            vear[i]->setVertex(1,VA2);
            Eigen::Matrix3d InfoA = pKFi->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
            vear[i]->setInformation(InfoA);
            optimizer.addEdge(vear[i]);
        }
        else
            Verbose::PrintMess("ERROR building inertial edge", Verbose::VERBOSITY_NORMAL);
    }

    Verbose::PrintMess("end inserting inertial edges", Verbose::VERBOSITY_NORMAL);


    // Set MapPoint vertices
    const int nExpectedSize = (N+Ncov+lFixedKeyFrames.size())*lLocalMapPoints.size();

    // Mono
    vector<EdgeMono*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    // Stereo
    vector<EdgeStereo*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float chi2Mono2 = 5.991;
    const float thHuberStereo = sqrt(7.815);
    const float chi2Stereo2 = 7.815;

    const unsigned long iniMPid = maxKFid*5;

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        if (!pMP)
            continue;

        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());

        unsigned long id = pMP->mnId+iniMPid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

        // Create visual constraints
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if (!pKFi)
                continue;

            if ((pKFi->mnBALocalForKF!=pCurrKF->mnId) && (pKFi->mnBAFixedForKF!=pCurrKF->mnId))
                continue;

            if (pKFi->mnId>maxKFid){
                continue;
            }


            if(optimizer.vertex(id)==NULL || optimizer.vertex(pKFi->mnId)==NULL)
                continue;

            if(!pKFi->isBad())
            {
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[get<0>(mit->second)];

                if(pKFi->mvuRight[get<0>(mit->second)]<0) // Monocular observation
                {
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMono* e = new EdgeMono();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);
                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // stereo observation
                {
                    const float kp_ur = pKFi->mvuRight[get<0>(mit->second)];
                    Eigen::Matrix<double,3,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereo* e = new EdgeStereo();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(8);

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

    // Check inlier observations
    // Mono
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        EdgeMono* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>chi2Mono2)
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Stereo
    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        EdgeStereo* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>chi2Stereo2)
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Get Map Mutex and erase outliers
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);
    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }


    // Recover optimized data
    //Keyframes
    for(int i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
        pKFi->SetPose(Tcw);

        Sophus::SE3d Tiw = pKFi->GetPose().cast<double>();
        g2o::Sim3 g2oSiw(Tiw.unit_quaternion(),Tiw.translation(),1.0);
        corrPoses[pKFi] = g2oSiw;

        if(pKFi->bImu)
        {
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
            pKFi->SetVelocity(VV->estimate().cast<float>());
            VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
            VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
            Vector6d b;
            b << VG->estimate(), VA->estimate();
            pKFi->SetNewBias(IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]));
        }
    }

    for(int i=0; i<Ncov; i++)
    {
        KeyFrame* pKFi = vpOptimizableCovKFs[i];

        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
        pKFi->SetPose(Tcw);

        Sophus::SE3d Tiw = pKFi->GetPose().cast<double>();
        g2o::Sim3 g2oSiw(Tiw.unit_quaternion(),Tiw.translation(),1.0);
        corrPoses[pKFi] = g2oSiw;

        if(pKFi->bImu)
        {
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
            pKFi->SetVelocity(VV->estimate().cast<float>());
            VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
            VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
            Vector6d b;
            b << VG->estimate(), VA->estimate();
            pKFi->SetNewBias(IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]));
        }
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+iniMPid+1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();
    }

    pMap->IncreaseChangeIndex();
}

int Optimizer::PoseInertialOptimizationLastKeyFrame(Frame *pFrame, bool bRecInit)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(solver);

    int nInitialMonoCorrespondences=0;
    int nInitialStereoCorrespondences=0;
    int nInitialCorrespondences=0;

    // Set Frame vertex
    VertexPose* VP = new VertexPose(pFrame);
    VP->setId(0);
    VP->setFixed(false);
    optimizer.addVertex(VP);
    VertexVelocity* VV = new VertexVelocity(pFrame);
    VV->setId(1);
    VV->setFixed(false);
    optimizer.addVertex(VV);
    VertexGyroBias* VG = new VertexGyroBias(pFrame);
    VG->setId(2);
    VG->setFixed(false);
    optimizer.addVertex(VG);
    VertexAccBias* VA = new VertexAccBias(pFrame);
    VA->setId(3);
    VA->setFixed(false);
    optimizer.addVertex(VA);

    // Set MapPoint vertices
    const int N = pFrame->N;
    const int Nleft = pFrame->Nleft;
    const bool bRight = (Nleft!=-1);

    vector<EdgeMonoOnlyPose*> vpEdgesMono;
    vector<EdgeStereoOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeMono;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesMono.reserve(N);
    vpEdgesStereo.reserve(N);
    vnIndexEdgeMono.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        for(int i=0; i<N; i++)
        {
            MapPoint* pMP = pFrame->mvpMapPoints[i];
            if(pMP)
            {
                cv::KeyPoint kpUn;

                // Left monocular observation
                if((!bRight && pFrame->mvuRight[i]<0) || i < Nleft)
                {
                    if(i < Nleft) // pair left-right
                        kpUn = pFrame->mvKeys[i];
                    else
                        kpUn = pFrame->mvKeysUn[i];

                    nInitialMonoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),0);

                    e->setVertex(0,VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                // Stereo observation
                else if(!bRight)
                {
                    nInitialStereoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    kpUn = pFrame->mvKeysUn[i];
                    const float kp_ur = pFrame->mvuRight[i];
                    Eigen::Matrix<double,3,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereoOnlyPose* e = new EdgeStereoOnlyPose(pMP->GetWorldPos());

                    e->setVertex(0, VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

                    const float &invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }

                // Right monocular observation
                if(bRight && i >= Nleft)
                {
                    nInitialMonoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    kpUn = pFrame->mvKeysRight[i - Nleft];
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),1);

                    e->setVertex(0,VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
            }
        }
    }
    // ---- spherical line edges: SAME as PoseInertialOptimizationLastFrame.
    // This optimizer previously had ZERO line edges, so the estimator
    // alternated between seeing and not seeing lines depending on which pose
    // path ran -- a joint point+line estimator must use both everywhere.
    int nLineEdgesK = 0;
    std::vector<EdgeLineOnlyPose*> vpEdgesLineK;
    std::vector<size_t> vnIndexEdgeLineK;
    if(MapLine::kVoteInPose && !pFrame->mvLines.empty() && !pFrame->mvpMapLines.empty())
    {
        const float f0 = pFrame->mpCamera ? pFrame->mpCamera->getParameter(0) : 400.f;
        const double infoLineK = 1.0 / 9.0;   // sigma = 3 px (residual is in PIXELS)
        for(size_t i = 0; i < pFrame->mvLines.size() && i < pFrame->mvpMapLines.size(); i++)
        {
            MapLine* pML = pFrame->mvpMapLines[i];
            if(!pML || pML->isBad() || pML->mnValidated < 2) continue;
            const LineObs &lo = pFrame->mvLines[i];
            Eigen::Vector3f eS, eE; pML->GetEndpoints(eS, eE);
            EdgeLineOnlyPose* el = new EdgeLineOnlyPose(
                eS, eE, lo.b1u, lo.b2u, f0, lo.cam, pML->mCreateParallax);
            el->setVertex(0, VP);
            el->setMeasurement(Eigen::Vector2d::Zero());
            el->setInformation(Eigen::Matrix2d::Identity() * infoLineK);
            g2o::RobustKernelHuber* rkl = new g2o::RobustKernelHuber;
            el->setRobustKernel(rkl);
            rkl->setDelta(sqrt(5.991));
            optimizer.addEdge(el);
            vpEdgesLineK.push_back(el);
            vnIndexEdgeLineK.push_back(i);
            nLineEdgesK++;
        }
    }

    nInitialCorrespondences = nInitialMonoCorrespondences + nInitialStereoCorrespondences;

    KeyFrame* pKF = pFrame->mpLastKeyFrame;
    VertexPose* VPk = new VertexPose(pKF);
    VPk->setId(4);
    VPk->setFixed(true);
    optimizer.addVertex(VPk);
    VertexVelocity* VVk = new VertexVelocity(pKF);
    VVk->setId(5);
    VVk->setFixed(true);
    optimizer.addVertex(VVk);
    VertexGyroBias* VGk = new VertexGyroBias(pKF);
    VGk->setId(6);
    VGk->setFixed(true);
    optimizer.addVertex(VGk);
    VertexAccBias* VAk = new VertexAccBias(pKF);
    VAk->setId(7);
    VAk->setFixed(true);
    optimizer.addVertex(VAk);

    EdgeInertial* ei = new EdgeInertial(pFrame->mpImuPreintegrated);

    ei->setVertex(0, VPk);
    ei->setVertex(1, VVk);
    ei->setVertex(2, VGk);
    ei->setVertex(3, VAk);
    ei->setVertex(4, VP);
    ei->setVertex(5, VV);
    optimizer.addEdge(ei);

    EdgeGyroRW* egr = new EdgeGyroRW();
    egr->setVertex(0,VGk);
    egr->setVertex(1,VG);
    Eigen::Matrix3d InfoG = pFrame->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
    egr->setInformation(InfoG);
    optimizer.addEdge(egr);

    EdgeAccRW* ear = new EdgeAccRW();
    ear->setVertex(0,VAk);
    ear->setVertex(1,VA);
    Eigen::Matrix3d InfoA = pFrame->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
    ear->setInformation(InfoA);
    optimizer.addEdge(ear);

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    float chi2Mono[4]={12,7.5,5.991,5.991};
    float chi2Stereo[4]={15.6,9.8,7.815,7.815};

    int its[4]={10,10,10,10};

    int nBad = 0;
    int nBadMono = 0;
    int nBadStereo = 0;
    int nInliersMono = 0;
    int nInliersStereo = 0;
    int nInliers = 0;
    for(size_t it=0; it<4; it++)
    {
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad = 0;
        nBadMono = 0;
        nBadStereo = 0;
        nInliers = 0;
        nInliersMono = 0;
        nInliersStereo = 0;
        float chi2close = 1.5*chi2Mono[it];

        // For monocular observations
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            EdgeMonoOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();
            bool bClose = pFrame->mvpMapPoints[idx]->mTrackDepth<10.f;

            if((chi2>chi2Mono[it]&&!bClose)||(bClose && chi2>chi2close)||!e->isDepthPositive())
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBadMono++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
                nInliersMono++;
            }

            if (it==2)
                e->setRobustKernel(0);
        }

        // For stereo observations
        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            EdgeStereoOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1); // not included in next optimization
                nBadStereo++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
                nInliersStereo++;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesLineK.size(); i<iend; i++)
        {
            EdgeLineOnlyPose* el = vpEdgesLineK[i];
            const size_t idx = vnIndexEdgeLineK[i];
            if(idx < pFrame->mvbLineOutlier.size() && pFrame->mvbLineOutlier[idx])
                el->computeError();
            const float chi2l = el->chi2();
            const bool bad = (chi2l > chi2Mono[it]) || !el->isDepthPositive();
            if(idx < pFrame->mvbLineOutlier.size())
                pFrame->mvbLineOutlier[idx] = bad;
            el->setLevel(bad ? 1 : 0);
            if(it == 2)
                el->setRobustKernel(0);
        }

        nInliers = nInliersMono + nInliersStereo;
        nBad = nBadMono + nBadStereo;

        if(optimizer.edges().size()<10)
        {
            break;
        }

    }

    // If not too much tracks, recover not too bad points
    if ((nInliers<30) && !bRecInit)
    {
        nBad=0;
        const float chi2MonoOut = 18.f;
        const float chi2StereoOut = 24.f;
        EdgeMonoOnlyPose* e1;
        EdgeStereoOnlyPose* e2;
        for(size_t i=0, iend=vnIndexEdgeMono.size(); i<iend; i++)
        {
            const size_t idx = vnIndexEdgeMono[i];
            e1 = vpEdgesMono[i];
            e1->computeError();
            if (e1->chi2()<chi2MonoOut)
                pFrame->mvbOutlier[idx]=false;
            else
                nBad++;
        }
        for(size_t i=0, iend=vnIndexEdgeStereo.size(); i<iend; i++)
        {
            const size_t idx = vnIndexEdgeStereo[i];
            e2 = vpEdgesStereo[i];
            e2->computeError();
            if (e2->chi2()<chi2StereoOut)
                pFrame->mvbOutlier[idx]=false;
            else
                nBad++;
        }
    }

    // Recover optimized pose, velocity and biases
    pFrame->SetImuPoseVelocity(VP->estimate().Rwb.cast<float>(), VP->estimate().twb.cast<float>(), VV->estimate().cast<float>());
    Vector6d b;
    b << VG->estimate(), VA->estimate();
    pFrame->mImuBias = IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]);

    // Recover Hessian, marginalize keyFframe states and generate new prior for frame
    Eigen::Matrix<double,15,15> H;
    H.setZero();

    H.block<9,9>(0,0)+= ei->GetHessian2();
    H.block<3,3>(9,9) += egr->GetHessian2();
    H.block<3,3>(12,12) += ear->GetHessian2();

    int tot_in = 0, tot_out = 0;
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
    {
        EdgeMonoOnlyPose* e = vpEdgesMono[i];

        const size_t idx = vnIndexEdgeMono[i];

        if(!pFrame->mvbOutlier[idx])
        {
            H.block<6,6>(0,0) += e->GetHessian();
            tot_in++;
        }
        else
            tot_out++;
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
    {
        EdgeStereoOnlyPose* e = vpEdgesStereo[i];

        const size_t idx = vnIndexEdgeStereo[i];

        if(!pFrame->mvbOutlier[idx])
        {
            H.block<6,6>(0,0) += e->GetHessian();
            tot_in++;
        }
        else
            tot_out++;
    }

    pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb,VP->estimate().twb,VV->estimate(),VG->estimate(),VA->estimate(),H);

    return nInitialCorrespondences-nBad;
}

int Optimizer::PoseInertialOptimizationLastFrame(Frame *pFrame, bool bRecInit)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    int nInitialMonoCorrespondences=0;
    int nInitialStereoCorrespondences=0;
    int nInitialCorrespondences=0;

    // Set Current Frame vertex
    VertexPose* VP = new VertexPose(pFrame);
    VP->setId(0);
    VP->setFixed(false);
    optimizer.addVertex(VP);
    VertexVelocity* VV = new VertexVelocity(pFrame);
    VV->setId(1);
    VV->setFixed(false);
    optimizer.addVertex(VV);
    VertexGyroBias* VG = new VertexGyroBias(pFrame);
    VG->setId(2);
    VG->setFixed(false);
    optimizer.addVertex(VG);
    VertexAccBias* VA = new VertexAccBias(pFrame);
    VA->setId(3);
    VA->setFixed(false);
    optimizer.addVertex(VA);

    // Set MapPoint vertices
    const int N = pFrame->N;
    const int Nleft = pFrame->Nleft;
    const bool bRight = (Nleft!=-1);

    vector<EdgeMonoOnlyPose*> vpEdgesMono;
    vector<EdgeStereoOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeMono;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesMono.reserve(N);
    vpEdgesStereo.reserve(N);
    vnIndexEdgeMono.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        // LINES-ONLY ISOLATION (Optimizer::bLinesOnly). Withhold every point
        // edge so the pose is fixed by IMU + line constraints alone. If the
        // line geometry is right this still tracks; if it diverges, the fault
        // is in the lines and not in some interaction with the points. Points
        // are still tracked and mapped -- only their EDGES are withheld here.
        for(int i=0; i<N && !bLinesOnly; i++)
        {
            MapPoint* pMP = pFrame->mvpMapPoints[i];
            if(pMP)
            {
                cv::KeyPoint kpUn;
                // Left monocular observation
                if((!bRight && pFrame->mvuRight[i]<0) || i < Nleft)
                {
                    if(i < Nleft) // pair left-right
                        kpUn = pFrame->mvKeys[i];
                    else
                        kpUn = pFrame->mvKeysUn[i];

                    nInitialMonoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),0);

                    e->setVertex(0,VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                // Stereo observation
                else if(!bRight)
                {
                    nInitialStereoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    kpUn = pFrame->mvKeysUn[i];
                    const float kp_ur = pFrame->mvuRight[i];
                    Eigen::Matrix<double,3,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereoOnlyPose* e = new EdgeStereoOnlyPose(pMP->GetWorldPos());

                    e->setVertex(0, VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

                    const float &invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }

                // Right monocular observation
                if(bRight && i >= Nleft)
                {
                    nInitialMonoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    kpUn = pFrame->mvKeysRight[i - Nleft];
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),1);

                    e->setVertex(0,VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
            }
        }
    }


    // ---- spherical line constraints ---------------------------------------
    // A line contributes n_c . b for each observed endpoint bearing, where
    //   n_c = Rcw * m + tcw x (Rcw * d)
    // is the great-circle normal of the world line (d,m) in THIS camera. The
    // camera index comes from the observation itself, so rear-lens lines are
    // projected through the rig's camera 1, never camera 0.
    //
    // Information: the residual is a SINE of an angle, so its noise is roughly
    // the angular uncertainty of an endpoint bearing -- about one pixel over
    // the focal length. sigma = 1/f, hence information = f^2.
    int nLineEdges = 0;
    std::vector<EdgeLineOnlyPose*> vpEdgesLine;
    std::vector<size_t> vnIndexEdgeLine;
    if(MapLine::kVoteInPose && !pFrame->mvLines.empty() && !pFrame->mvpMapLines.empty())
    {
        const float f0 = pFrame->mpCamera ? pFrame->mpCamera->getParameter(0) : 400.f;
        const double infoLine = 1.0;          // residual is in PIXELS
        for(size_t i = 0; i < pFrame->mvLines.size() && i < pFrame->mvpMapLines.size(); i++)
        {
            MapLine* pML = pFrame->mvpMapLines[i];
            if(!pML || pML->isBad()) continue;
            const LineObs &lo = pFrame->mvLines[i];
            if(pML->mnValidated < 2) continue;   // FIX 4: probation (below)
            Eigen::Vector3f eS, eE; pML->GetEndpoints(eS, eE);
            EdgeLineOnlyPose* el = new EdgeLineOnlyPose(
                eS, eE, lo.b1u, lo.b2u, f0, lo.cam, pML->mCreateParallax);
            el->setVertex(0, VP);
            el->setMeasurement(Eigen::Vector2d::Zero());
            // FIX 3: the landmark is NOT noiseless. Its plane was triangulated
            // from two sigma_b-noisy planes meeting at angle theta_p, so the
            // residual's variance is sigma_b^2*(1 + 1/sin^2(theta_p)):
            //   info = f^2 * s^2/(1+s^2),   s = sin(creation parallax).
            // The old flat f^2 asserted pixel-perfect landmarks, which made a
            // 2-deg-parallax line pull as hard as a well-conditioned one --
            // measured as whitened chi2 ~13 even for HONEST lines (points: ~3).
            // Calibrated empirically, not derived: under info=f^2 the HONEST
            // lines (validated, low-parallax bucket) ran at whitened chi2
            // median ~13 where a correct 2-dof model gives 1.386 -- so the
            // real angular noise is ~3 px equivalent, roughly parallax-flat
            // (the parallax-dependent tail is mismatches, which probation,
            // cheirality and the chi2 rounds handle). sigma = 3 px:
            const double infoThis = infoLine / 9.0;
            el->setInformation(Eigen::Matrix2d::Identity() * infoThis);
            g2o::RobustKernelHuber* rkl = new g2o::RobustKernelHuber;
            el->setRobustKernel(rkl);
            // g2o compares delta against the CHI2 (residual x information), not
            // against the raw residual. With information = f^2 the old 0.02 --
            // written intending "1.1 deg" -- actually meant 0.009 PIXELS, so
            // every line sat in Huber's saturated branch where the gradient
            // magnitude is constant: a badly wrong line pulled exactly as hard
            // as a perfect one, and ~2000 of them per frame outvoted the point
            // inliers. Lines carry no scale information, so outvoting the IMU
            // that way collapsed metric scale (path length 2x, scale 0.12).
            // sqrt(5.991) is the 2-DoF chi2 95% threshold -- the same
            // convention the point edges use, so lines and points are now
            // statistically comparable.
            rkl->setDelta(sqrt(5.991));
            optimizer.addEdge(el);
            vpEdgesLine.push_back(el);
            vnIndexEdgeLine.push_back(i);
            nLineEdges++;
        }
    }

    {   // PROOF: how many line edges does the pose optimizer actually receive?
        static long nCalls = 0, nEdgesTotal = 0, nCallsWithLines = 0;
        nCalls++; nEdgesTotal += nLineEdges;
        if(nLineEdges > 0) nCallsWithLines++;
        if(nCalls % 100 == 0)
            std::cout << "[Debug] LineEdges: " << nCallsWithLines << "/" << nCalls
                      << " optimizer calls had ANY line edge; " << nEdgesTotal
                      << " line edges total" << std::endl;
    }

    nInitialCorrespondences = nInitialMonoCorrespondences + nInitialStereoCorrespondences;
    if(bLinesOnly)
        nInitialCorrespondences += nLineEdges;   // matched by nInliersLine below

    // Set Previous Frame Vertex
    Frame* pFp = pFrame->mpPrevFrame;

    VertexPose* VPk = new VertexPose(pFp);
    VPk->setId(4);
    VPk->setFixed(false);
    optimizer.addVertex(VPk);
    VertexVelocity* VVk = new VertexVelocity(pFp);
    VVk->setId(5);
    VVk->setFixed(false);
    optimizer.addVertex(VVk);
    VertexGyroBias* VGk = new VertexGyroBias(pFp);
    VGk->setId(6);
    VGk->setFixed(false);
    optimizer.addVertex(VGk);
    VertexAccBias* VAk = new VertexAccBias(pFp);
    VAk->setId(7);
    VAk->setFixed(false);
    optimizer.addVertex(VAk);

    EdgeInertial* ei = new EdgeInertial(pFrame->mpImuPreintegratedFrame);

    ei->setVertex(0, VPk);
    ei->setVertex(1, VVk);
    ei->setVertex(2, VGk);
    ei->setVertex(3, VAk);
    ei->setVertex(4, VP);
    ei->setVertex(5, VV);
    optimizer.addEdge(ei);

    EdgeGyroRW* egr = new EdgeGyroRW();
    egr->setVertex(0,VGk);
    egr->setVertex(1,VG);
    Eigen::Matrix3d InfoG = pFrame->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
    egr->setInformation(InfoG);
    optimizer.addEdge(egr);

    EdgeAccRW* ear = new EdgeAccRW();
    ear->setVertex(0,VAk);
    ear->setVertex(1,VA);
    Eigen::Matrix3d InfoA = pFrame->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
    ear->setInformation(InfoA);
    optimizer.addEdge(ear);

    if (!pFp->mpcpi)
        Verbose::PrintMess("pFp->mpcpi does not exist!!!\nPrevious Frame " + to_string(pFp->mnId), Verbose::VERBOSITY_NORMAL);

    EdgePriorPoseImu* ep = new EdgePriorPoseImu(pFp->mpcpi);

    ep->setVertex(0,VPk);
    ep->setVertex(1,VVk);
    ep->setVertex(2,VGk);
    ep->setVertex(3,VAk);
    g2o::RobustKernelHuber* rkp = new g2o::RobustKernelHuber;
    ep->setRobustKernel(rkp);
    rkp->setDelta(5);
    optimizer.addEdge(ep);

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={15.6f,9.8f,7.815f,7.815f};
    const int its[4]={10,10,10,10};

    int nBad=0;
    int nBadMono = 0;
    int nBadStereo = 0;
    int nInliersMono = 0;
    int nInliersStereo = 0;
    int nInliers=0;
    for(size_t it=0; it<4; it++)
    {
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad=0;
        nBadMono = 0;
        nBadStereo = 0;
        nInliers=0;
        nInliersMono=0;
        nInliersStereo=0;
        float chi2close = 1.5*chi2Mono[it];

        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            EdgeMonoOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];
            bool bClose = pFrame->mvpMapPoints[idx]->mTrackDepth<10.f;

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if((chi2>chi2Mono[it]&&!bClose)||(bClose && chi2>chi2close)||!e->isDepthPositive())
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBadMono++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
                nInliersMono++;
            }

            if (it==2)
                e->setRobustKernel(0);

        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            EdgeStereoOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBadStereo++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
                nInliersStereo++;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        // LINES GET THE SAME DISCIPLINE AS POINTS.
        //
        // Point edges are chi2-tested after every round and excluded when they
        // disagree (setLevel(1)), and at it==2 they lose their robust kernel so
        // the survivors are fit hard. Line edges used to get none of that: they
        // stayed at level 0 for all four rounds however wrong they were, and
        // kept their kernel throughout.
        //
        // That asymmetry is what destroyed the combination. Round 1, a bad line
        // shifts the pose; the points that disagree with that shifted pose blow
        // past chi2 and are dropped; rounds 2-4 run on whatever points still
        // agree with the lines. The lines became an unfilterable prior that
        // pruned the points into agreeing with them -- which is why points
        // alone worked, lines alone worked, and together they diverged.
        //
        // The residual is 2-DoF like a monocular point, so it takes the same
        // 5.991 threshold.
        int nBadLine = 0, nInliersLine = 0;
        for(size_t i=0, iend=vpEdgesLine.size(); i<iend; i++)
        {
            EdgeLineOnlyPose* el = vpEdgesLine[i];
            const size_t idx = vnIndexEdgeLine[i];
            if(idx < pFrame->mvbLineOutlier.size() && pFrame->mvbLineOutlier[idx])
                el->computeError();
            const float chi2l = el->chi2();
            const bool bad = (chi2l > chi2Mono[it]) || !el->isDepthPositive();
            if(idx < pFrame->mvbLineOutlier.size())
                pFrame->mvbLineOutlier[idx] = bad;
            el->setLevel(bad ? 1 : 0);
            if(bad) nBadLine++; else nInliersLine++;
            if(it == 2)
                el->setRobustKernel(0);
        }
        if(bLinesOnly)
        {
            nInliersMono += nInliersLine;      // else the caller reads 0 inliers
            nBadMono     += nBadLine;
        }

        nInliers = nInliersMono + nInliersStereo;
        nBad = nBadMono + nBadStereo;

        if(optimizer.edges().size()<10)
        {
            break;
        }
    }


    if ((nInliers<30) && !bRecInit)
    {
        nBad=0;
        const float chi2MonoOut = 18.f;
        const float chi2StereoOut = 24.f;
        EdgeMonoOnlyPose* e1;
        EdgeStereoOnlyPose* e2;
        for(size_t i=0, iend=vnIndexEdgeMono.size(); i<iend; i++)
        {
            const size_t idx = vnIndexEdgeMono[i];
            e1 = vpEdgesMono[i];
            e1->computeError();
            if (e1->chi2()<chi2MonoOut)
                pFrame->mvbOutlier[idx]=false;
            else
                nBad++;

        }
        for(size_t i=0, iend=vnIndexEdgeStereo.size(); i<iend; i++)
        {
            const size_t idx = vnIndexEdgeStereo[i];
            e2 = vpEdgesStereo[i];
            e2->computeError();
            if (e2->chi2()<chi2StereoOut)
                pFrame->mvbOutlier[idx]=false;
            else
                nBad++;
        }
    }

    {   // [AUDIT-E] after optimization: whitened chi2 by SOURCE. If rear-lens
        // (cam 1) line edges are systematically worse than front, the rig
        // transform inside the vertex (Rcw[1] via Trl) is wrong; if BOTH are
        // huge next to points, the landmark geometry is wrong.
        static std::vector<double> c0, c1, cp; static long nCall = 0;
        static std::vector<double> bk[4];   // parallax buckets: <3, 3-5, 5-10, >10 deg
        for(size_t i=0; i<vpEdgesLine.size(); i++){
            vpEdgesLine[i]->computeError();
            (vpEdgesLine[i]->cam_idx == 0 ? c0 : c1).push_back(vpEdgesLine[i]->chi2());
            const double pdeg = 180.0/M_PI * vpEdgesLine[i]->parallax;
            bk[pdeg < 3 ? 0 : pdeg < 5 ? 1 : pdeg < 10 ? 2 : 3].push_back(vpEdgesLine[i]->chi2());
        }
        for(size_t i=0; i<vpEdgesMono.size(); i++){
            vpEdgesMono[i]->computeError();
            cp.push_back(vpEdgesMono[i]->chi2());
        }
        if(++nCall % 200 == 0){
            auto med=[](std::vector<double>& v)->double{
                if(v.empty()) return -1;
                std::nth_element(v.begin(), v.begin()+v.size()/2, v.end());
                return v[v.size()/2]; };
            std::cout << "[AUDIT] optimizer chi2 median -- POINT " << med(cp)
                      << " | LINE cam0 " << med(c0) << " (" << c0.size()
                      << ") | LINE cam1 " << med(c1) << " (" << c1.size() << ")"
                      << std::endl;
            std::cout << "[AUDIT] line chi2 by CREATION PARALLAX -- <3deg "
                      << med(bk[0]) << " (" << bk[0].size() << ") | 3-5 "
                      << med(bk[1]) << " (" << bk[1].size() << ") | 5-10 "
                      << med(bk[2]) << " (" << bk[2].size() << ") | >10 "
                      << med(bk[3]) << " (" << bk[3].size() << ")" << std::endl;
            c0.clear(); c1.clear(); cp.clear();
            for(auto& b : bk) b.clear();
        }
    }

    nInliers = nInliersMono + nInliersStereo;

    // A line that never came back inside chi2 is not this line: forget the
    // association so the next frame re-triangulates instead of inheriting it.
    for(size_t i=0, iend=vnIndexEdgeLine.size(); i<iend; i++)
    {
        const size_t idx = vnIndexEdgeLine[i];
        if(idx < pFrame->mvbLineOutlier.size() && pFrame->mvbLineOutlier[idx]
           && idx < pFrame->mvpMapLines.size())
            pFrame->mvpMapLines[idx] = static_cast<MapLine*>(NULL);
    }


    // Recover optimized pose, velocity and biases
    pFrame->SetImuPoseVelocity(VP->estimate().Rwb.cast<float>(), VP->estimate().twb.cast<float>(), VV->estimate().cast<float>());
    Vector6d b;
    b << VG->estimate(), VA->estimate();
    pFrame->mImuBias = IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]);

    // Recover Hessian, marginalize previous frame states and generate new prior for frame
    Eigen::Matrix<double,30,30> H;
    H.setZero();

    H.block<24,24>(0,0)+= ei->GetHessian();

    Eigen::Matrix<double,6,6> Hgr = egr->GetHessian();
    H.block<3,3>(9,9) += Hgr.block<3,3>(0,0);
    H.block<3,3>(9,24) += Hgr.block<3,3>(0,3);
    H.block<3,3>(24,9) += Hgr.block<3,3>(3,0);
    H.block<3,3>(24,24) += Hgr.block<3,3>(3,3);

    Eigen::Matrix<double,6,6> Har = ear->GetHessian();
    H.block<3,3>(12,12) += Har.block<3,3>(0,0);
    H.block<3,3>(12,27) += Har.block<3,3>(0,3);
    H.block<3,3>(27,12) += Har.block<3,3>(3,0);
    H.block<3,3>(27,27) += Har.block<3,3>(3,3);

    H.block<15,15>(0,0) += ep->GetHessian();

    int tot_in = 0, tot_out = 0;
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
    {
        EdgeMonoOnlyPose* e = vpEdgesMono[i];

        const size_t idx = vnIndexEdgeMono[i];

        if(!pFrame->mvbOutlier[idx])
        {
            H.block<6,6>(15,15) += e->GetHessian();
            tot_in++;
        }
        else
            tot_out++;
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
    {
        EdgeStereoOnlyPose* e = vpEdgesStereo[i];

        const size_t idx = vnIndexEdgeStereo[i];

        if(!pFrame->mvbOutlier[idx])
        {
            H.block<6,6>(15,15) += e->GetHessian();
            tot_in++;
        }
        else
            tot_out++;
    }

    H = Marginalize(H,0,14);

    pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb,VP->estimate().twb,VV->estimate(),VG->estimate(),VA->estimate(),H.block<15,15>(15,15));
    delete pFp->mpcpi;
    pFp->mpcpi = NULL;

    return nInitialCorrespondences-nBad;
}

void Optimizer::OptimizeEssentialGraph4DoF(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections)
{
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<4, 4> > BlockSolver_4_4;

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolverX::LinearSolverType * linearSolver =
            new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    optimizer.setAlgorithm(solver);

    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);

    vector<VertexPose4DoF*> vpVertices(nMaxKFid+1);

    const int minFeat = 100;
    // Set KeyFrame vertices
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;

        VertexPose4DoF* V4DoF;

        const int nIDi = pKF->mnId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        if(it!=CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            const g2o::Sim3 Swc = it->second.inverse();
            Eigen::Matrix3d Rwc = Swc.rotation().toRotationMatrix();
            Eigen::Vector3d twc = Swc.translation();
            V4DoF = new VertexPose4DoF(Rwc, twc, pKF);
        }
        else
        {
            Sophus::SE3d Tcw = pKF->GetPose().cast<double>();
            g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);

            vScw[nIDi] = Siw;
            V4DoF = new VertexPose4DoF(pKF);
        }

        if(pKF==pLoopKF)
            V4DoF->setFixed(true);

        V4DoF->setId(nIDi);
        V4DoF->setMarginalized(false);

        optimizer.addVertex(V4DoF);
        vpVertices[nIDi]=V4DoF;
    }
    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    // Edge used in posegraph has still 6Dof, even if updates of camera poses are just in 4DoF
    Eigen::Matrix<double,6,6> matLambda = Eigen::Matrix<double,6,6>::Identity();
    matLambda(0,0) = 1e3;
    matLambda(1,1) = 1e3;
    matLambda(0,0) = 1e3;

    // Set Loop edges
    Edge4DoF* e_loop;
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];

        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sij = Siw * Sjw.inverse();
            Eigen::Matrix4d Tij;
            Tij.block<3,3>(0,0) = Sij.rotation().toRotationMatrix();
            Tij.block<3,1>(0,3) = Sij.translation();
            Tij(3,3) = 1.;

            Edge4DoF* e = new Edge4DoF(Tij);
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));

            e->information() = matLambda;
            e_loop = e;
            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // 1. Set normal edges
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Siw;

        // Use noncorrected poses for posegraph edges
        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if(iti!=NonCorrectedSim3.end())
            Siw = iti->second;
        else
            Siw = vScw[nIDi];

        // 1.1.0 Spanning tree edge
        KeyFrame* pParentKF = static_cast<KeyFrame*>(NULL);
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Swj;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if(itj!=NonCorrectedSim3.end())
                Swj = (itj->second).inverse();
            else
                Swj =  vScw[nIDj].inverse();

            g2o::Sim3 Sij = Siw * Swj;
            Eigen::Matrix4d Tij;
            Tij.block<3,3>(0,0) = Sij.rotation().toRotationMatrix();
            Tij.block<3,1>(0,3) = Sij.translation();
            Tij(3,3)=1.;

            Edge4DoF* e = new Edge4DoF(Tij);
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // 1.1.1 Inertial edges
        KeyFrame* prevKF = pKF->mPrevKF;
        if(prevKF)
        {
            int nIDj = prevKF->mnId;

            g2o::Sim3 Swj;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(prevKF);

            if(itj!=NonCorrectedSim3.end())
                Swj = (itj->second).inverse();
            else
                Swj =  vScw[nIDj].inverse();

            g2o::Sim3 Sij = Siw * Swj;
            Eigen::Matrix4d Tij;
            Tij.block<3,3>(0,0) = Sij.rotation().toRotationMatrix();
            Tij.block<3,1>(0,3) = Sij.translation();
            Tij(3,3)=1.;

            Edge4DoF* e = new Edge4DoF(Tij);
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // 1.2 Loop edges
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(pLKF->mnId<pKF->mnId)
            {
                g2o::Sim3 Swl;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if(itl!=NonCorrectedSim3.end())
                    Swl = itl->second.inverse();
                else
                    Swl = vScw[pLKF->mnId].inverse();

                g2o::Sim3 Sil = Siw * Swl;
                Eigen::Matrix4d Til;
                Til.block<3,3>(0,0) = Sil.rotation().toRotationMatrix();
                Til.block<3,1>(0,3) = Sil.translation();
                Til(3,3) = 1.;

                Edge4DoF* e = new Edge4DoF(Til);
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                e->information() = matLambda;
                optimizer.addEdge(e);
            }
        }

        // 1.3 Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && pKFn!=prevKF && pKFn!=pKF->mNextKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Swn;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if(itn!=NonCorrectedSim3.end())
                        Swn = itn->second.inverse();
                    else
                        Swn = vScw[pKFn->mnId].inverse();

                    g2o::Sim3 Sin = Siw * Swn;
                    Eigen::Matrix4d Tin;
                    Tin.block<3,3>(0,0) = Sin.rotation().toRotationMatrix();
                    Tin.block<3,1>(0,3) = Sin.translation();
                    Tin(3,3) = 1.;
                    Edge4DoF* e = new Edge4DoF(Tin);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    e->information() = matLambda;
                    optimizer.addEdge(e);
                }
            }
        }
    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        VertexPose4DoF* Vi = static_cast<VertexPose4DoF*>(optimizer.vertex(nIDi));
        Eigen::Matrix3d Ri = Vi->estimate().Rcw[0];
        Eigen::Vector3d ti = Vi->estimate().tcw[0];

        g2o::Sim3 CorrectedSiw = g2o::Sim3(Ri,ti,1.);
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();

        Sophus::SE3d Tiw(CorrectedSiw.rotation(),CorrectedSiw.translation());
        pKFi->SetPose(Tiw.cast<float>());
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;

        KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
        nIDr = pRefKF->mnId;

        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        Eigen::Matrix<double,3,1> eigP3Dw = pMP->GetWorldPos().cast<double>();
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
        pMP->SetWorldPos(eigCorrectedP3Dw.cast<float>());

        pMP->UpdateNormalAndDepth();
    }
    pMap->IncreaseChangeIndex();
}

} //namespace ORB_SLAM

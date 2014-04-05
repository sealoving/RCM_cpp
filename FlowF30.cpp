#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <assert.h>
#include <exception>
#include <algorithm>
#include <time.h>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;
using namespace std;

// This file simulates flow of river delta.

#define SQUARE(x) (x.cwiseProduct(x))

// Reads a csv file and returns a matrix
MatrixXf readCSV(string filename) {
    ifstream infile(filename.c_str());
    if (infile.bad()) throw exception();
    
    // Rows and columns in the matrix
    int rows = 0, cols = -1;

    string line;
    vector<string> strs;
    while (getline(infile, line)) {
        boost::split(strs, line, boost::is_any_of(","));
        if (cols == -1)
            cols = strs.size();
        else assert(strs.size() == cols);
        rows++;
    }
    assert(rows > 0);
    assert(cols > 0);

    MatrixXf m (rows,cols);

    // Second pass
    infile.clear();
    infile.seekg(0, ios::beg);
    int row = 0;
    while (getline(infile, line)) {
        boost::split(strs, line, boost::is_any_of(","));
        for (int col=0; col<strs.size(); col++) {
            m(row,col) = boost::lexical_cast<float>(strs[col]);
        }
        row++;
    }
    infile.close();

    return m;
}


int L; // grid size, dip direction
int W; // grid size, strike direction

// Neighbor grid
// 321
// 4x0
// 567

// Neighbor vectors
vector<int> centerNeighbor      { 0,1,2,3,4,5,6,7 };
vector<int> leftNeighbor        { 0,1,2,6,7 };
vector<int> rightNeighbor       { 2,3,4,5,6 };
vector<int> topNeighbor         { 0,4,5,6,7 };
vector<int> bottomNeighbor      { 0,1,2,3,4 };
vector<int> topLeftNeighbor     { 0,6,7 };
vector<int> topRightNeighbor    { 4,5,6 };
vector<int> bottomLeftNeighbor  { 0,1,2 };
vector<int> bottomRightNeighbor { 2,3,4 };

vector<int>& nbr(int i, int j) {
    assert(i >= 0 && i < L);
    assert(j >= 0 && j < W);

    if (i == 0)
        if (j == W-1)
            return topLeftNeighbor;
        else if (j == 0)
            return bottomLeftNeighbor;
        else
            return leftNeighbor;
    else if (i == L-1)
        if (j == 0)
            return bottomRightNeighbor;
        else if (j == W-1)
            return topRightNeighbor;
        else
            return rightNeighbor;
    else
        if (j == 0)
            return bottomNeighbor;
        else if (j == W-1)
            return topNeighbor;
        else
            return centerNeighbor;
}

int main() {
    clock_t startTime = clock();

    // Seed to time for now...
    srand(time(NULL));

    MatrixXf eta = readCSV("topofile.csv");

    L = eta.rows(); // grid size, dip direction
    W = eta.cols(); // grid size, strike direction
    float dx = 50.; // (m)

    float u_ref = 1.0; // m/s, only used for surface calculation, to identify shoreline
    float u_max = 5*u_ref; //m/s, maximum velocity allowed in the calculation
    float h_dry = 0.025; //m, threshold depth for wet/dry transition
    float Cf = 0.01; // estimated coefficient of friction (constant for all cells)
    float H_SL = 0; // downstream sea-level
    float Q_water = 1250; //m^3/s, WLD total discharge (low)

    MatrixXi pxy_start(6,2);
    pxy_start << 1,58, 1,59, 1,60, 1,61, 1,62, 1,63;

    MatrixXf qx_guess = readCSV("RCMqx0.csv");
    MatrixXf qy_guess = readCSV("RCMqy0.csv");
    MatrixXf H_guess  = readCSV("RCMsfc0.csv");

    // initial depth
    MatrixXf h_guess  = (H_guess-eta).cwiseMax(MatrixXf::Zero(L,W)); 

    MatrixXi wall_flag  = readCSV("RCMwallF30.csv").cast<int>();

    // set downstream boundary
    MatrixXi boundflag = MatrixXi::Zero(L,W);
    boundflag.col(0)   = RowVectorXi::Ones(L);
    boundflag.col(W-1) = RowVectorXi::Ones(L);
    boundflag.row(0)   = RowVectorXi::Zero(W);
    boundflag.row(L-1) = RowVectorXi::Ones(W);
    
    // calculate the probability distribution among input cells
    RowVectorXf P_start(pxy_start.rows());
    for (int k=0; k<pxy_start.rows(); k++) {
        P_start(k) = h_guess(pxy_start(k,0), pxy_start(k,1));
    }
    P_start /= P_start.sum();

    // TODO: Plot

    float gamma = 0.05;
    static const float GRAVITY = 9.81;
    float omega_sfc = 0.1;
    float omega_flow = 0.1;
    int Nsmooth = 10;
    float Csmooth = 0.9;

    // ===========    preparation for random walks     ===========
    int Np_water = 200;
    float Qp_water = Q_water/float(Np_water)/dx;

    RowVectorXf dxn_ivec(8);
    dxn_ivec << 1,sqrt(0.5),0,-sqrt(0.5),-1,-sqrt(0.5),0,sqrt(0.5); // E --> clockwise
    RowVectorXf dxn_jvec(8);
    dxn_jvec << 0,sqrt(0.5),1,sqrt(0.5),0,-sqrt(0.5),-1,-sqrt(0.5); // E --> clockwise
    RowVectorXi dxn_iwalk(8);
    dxn_iwalk << 1,1,0,-1,-1,-1,0,1; // E --> clockwise
    RowVectorXi dxn_jwalk(8);
    dxn_jwalk << 0,1,1,1,0,-1,-1,-1; // E --> clockwise
    RowVectorXf dxn_dist(8);
    dxn_dist << 1,sqrt(2),1,sqrt(2),1,sqrt(2),1,sqrt(2);  // E --> clockwise
    // ===========================================================

    int itmax = 2*(L+W);
    MatrixXf qx = qx_guess;
    MatrixXf qy = qy_guess;
    MatrixXf qw = (SQUARE(qx)+SQUARE(qy)).cwiseSqrt();
    MatrixXf qxn = MatrixXf::Zero(L,W);
    MatrixXf qyn = MatrixXf::Zero(L,W);
    MatrixXf qwn = MatrixXf::Zero(L,W);
    MatrixXi prepath_flag = MatrixXi::Zero(L,W);

    MatrixXf H         = H_guess;
    MatrixXf Hnew      = MatrixXf::Zero(L,W);
    MatrixXf h         = h_guess;
    MatrixXi sfc_visit = MatrixXi::Zero(L,W);
    MatrixXf sfc_sum   = MatrixXf::Zero(L,W);
    MatrixXf Htemp     = MatrixXf::Zero(L,W);
    MatrixXf Hsmth     = MatrixXf::Zero(L,W);
    MatrixXf uw_avg    = MatrixXf::Zero(L,W);
    MatrixXf ux_avg    = MatrixXf::Zero(L,W);
    MatrixXf uy_avg    = MatrixXf::Zero(L,W);
    MatrixXf sfc_avg   = MatrixXf::Zero(L,W);
    MatrixXi wet_flag  = MatrixXi::Zero(L,W);
    MatrixXf uw        = MatrixXf::Zero(L,W);
    MatrixXf ux        = MatrixXf::Zero(L,W);
    MatrixXf uy        = MatrixXf::Zero(L,W);
    for (int i=0; i<L; i++) {
        for (int j=0; j<W; j++) {
            if (h(i,j) > h_dry) {
                uw(i,j) = min(u_max,qw(i,j)/h(i,j));
                ux(i,j) = uw(i,j)*qx(i,j)/qw(i,j);
                uy(i,j) = uw(i,j)*qy(i,j)/qw(i,j);
            }
        }
    }

    int pxn, pyn; // TODO: Scope of these vars?
    int istep, jstep; // TODO: Scope of these vars?

    int maxiter = 300;
    for (int iter=0; iter<maxiter; iter++) {
        cout << iter << endl;

        qxn.fill(0); qyn.fill(0); qwn.fill(0); wet_flag.fill(0);
        
        for (int i=0; i<L; i++) {
            for (int j=0; j<W; j++) {
                if (h(i,j) >= h_dry && wall_flag(i,j) == 0)
                    wet_flag(i,j) = 1;
            }
        }

        Hnew = eta.cwiseMax(MatrixXf::Constant(L,W,H_SL));
        Hnew = Hnew.cwiseProduct(MatrixXf::Constant(L,W,1)-boundflag.cast<float>());
        sfc_visit.fill(0);
        sfc_sum.fill(0);
        
        // Loop through the parcels
        for (int np=0; np<Np_water; np++) {
            prepath_flag.fill(0);

            // Choose which starting position to initialize the parcel
            float step_rand = double(rand()) / RAND_MAX;
            float C_start = P_start(0);
            int k=0;
            while (C_start < step_rand) {
                k++;
                C_start += P_start(k);
            }
            int px = pxy_start(k,0);
            int py = pxy_start(k,1);

            // assuming dxn_iwalk(1) and dxn_jwalk(1) gives the initial direction
            qxn(px,py) = qxn(px,py) + dxn_iwalk(0)/dxn_dist(0);
            qyn(px,py) = qyn(px,py) + dxn_jwalk(0)/dxn_dist(0);
            qwn(px,py) = qwn(px,py) + Qp_water/2;

            int it = 0; // keep a record of the walk path
            RowVectorXi iseq(2*(L+W));
            RowVectorXi jseq(2*(L+W));            
            iseq(it) = px;
            jseq(it) = py;

            bool water_continue = true; // flag for whether to continue walk

            while (water_continue && ++it < itmax) {
                prepath_flag(px,py) = 1;
                
                // get local out dxns 1:k
                vector<int> dxn = nbr(px,py);
                int nk = dxn.size();
                RowVectorXf weight(nk);  // TODO: These slow us down
                RowVectorXf weightsfc(nk); 
                RowVectorXf weightdxn(nk); 

                // calculate weightsfc and weightdxn
                for (int k=0; k<nk; k++) {
                    pxn = px+dxn_iwalk(dxn[k]);
                    pyn = py+dxn_jwalk(dxn[k]);
                    float dist = dxn_dist(dxn[k]);

                    if (wet_flag(pxn,pyn) == 1) {
                        weightsfc(k) = std::max(0.f,H(px,py)-H(pxn,pyn))/dist;
                        weightdxn(k) = std::max(0.f,qx(px,py)*dxn_ivec(dxn[k])+qy(px,py)*dxn_jvec(dxn[k]))/dist;
                    }
                }

                // normalize and calculate weight
                if (weightsfc.sum() != 0) {
                    // require that weightsfc >= 0
                    weightsfc = weightsfc/weightsfc.sum();
                }
                if (weightdxn.sum() != 0) {
                    // require that weightdxn >= 0
                    weightdxn = weightdxn/weightdxn.sum();
                }
                weight = gamma*weightsfc + weightdxn;
                for (int k=0; k<nk; k++) {
                    pxn = px+dxn_iwalk(dxn[k]);
                    pyn = py+dxn_jwalk(dxn[k]);
                    float dist = dxn_dist(dxn[k]);

                    if (wet_flag(pxn,pyn) == 1) {
                        weight(k) = pow(h(pxn,pyn),1.0*weight(k));
                    }
                }


                // if weight is not all zero
                if (weight.sum() != 0) {
                    weight = weight/weight.sum();
                    // choose target cell by probability
                    float weightSum = 0;
                    step_rand = 1-(double(rand()) / RAND_MAX);
                    for (int k=0; k<nk; k++) {
                        weightSum += weight(k);
                        if (step_rand < weightSum) {
                            istep = dxn_iwalk(dxn[k]);
                            jstep = dxn_jwalk(dxn[k]);
                            break;
                        }
                    }
                }
                // if weight is all zero
                if (weight.sum() == 0) { // TODO: Elseif?
                    pxn = max(1,px + (rand() % 3 - 1)); // TODO: What is the scope of pxn/pyn?
                    pxn = min(pxn,L);
                    int ntry = 0;
                    while (wet_flag(pxn,pyn) == 0 && ntry < 8) {
                        ntry++;
                        pxn = max(1,px+(rand() % 3 - 1));
                        pxn = min(pxn,L);
                        pyn = max(1,py+(rand()%3-1));
                        pyn = min(pyn,W);
                    }
                    istep = pxn-px;
                    jstep = pyn-py;
                }

                pxn = max(0,min(L-1,px+istep)); // TODO: Added this bounding...
                pyn = max(0,min(W-1,py+jstep));
                float dist = sqrt(pow(istep,2)+pow(jstep,2));
                if (dist > 0) {
                    qxn(px,py) = qxn(px,py)+istep/dist;
                    qyn(px,py) = qyn(px,py)+jstep/dist;
                    qwn(px,py) = qwn(px,py)+Qp_water/2;
                    qxn(pxn,pyn) = qxn(pxn,pyn)+istep/dist;
                    qyn(pxn,pyn) = qyn(pxn,pyn)+jstep/dist;
                    qwn(pxn,pyn) = qwn(pxn,pyn)+Qp_water/2;
                }

                px = pxn; // TODO: Causes assert error
                py = pyn;
                iseq(it) = px;
                jseq(it) = py;

                if (boundflag(px,py) == 1) {
                    water_continue = 0;
                    int itend = it; //TODO: Not used?
                }
                if (prepath_flag(px,py) == 1) {
                    px += round(((double(rand())/RAND_MAX)-0.5)*5);
                    py += round(((double(rand())/RAND_MAX)-0.5)*5);
                    px = max(px,6); px = min(L-1,px);
                    py = max(2,py); py = min(W-1,py);
                }
            }
            float dist = sqrt(istep*istep+jstep*jstep);
            if (dist > 0) {
                qxn(pxn,pyn) = qxn(pxn,pyn)+istep/dist;
                qyn(pxn,pyn) = qyn(pxn,pyn)+jstep/dist;
                qwn(pxn,pyn) = qwn(pxn,pyn)+Qp_water/2;
            }

            // calcuate free surface 
            // along a single water parcel path
            // values are stored for later update when all parcels are done
            float itback = iseq.cols()-1;
            if (boundflag(iseq(itback),jseq(itback)) == 1) { // && sfccredit == 1
                Hnew(iseq(itback),jseq(itback)) = H_SL;
                int it0 = 0;
                int Ldist = 0; //TODO: Not used?
                for (it=itback-1; it>=0; it--) { //TODO: Changed here itback-1. Should be -0
                    int i = iseq(it);
                    int ip = iseq(it+1);
                    int j = jseq(it);
                    int jp = jseq(it+1);
                    float dist = pow(pow(ip-i,2)+pow(jp-j,2),0.5);
                    float dH;
                    
                    if (dist > 0) {
                        if (it0 == 0) {
                            if (uw(i,j) > 0.1*u_ref || h(i,j) < 1.0) { // see if it is shoreline
                                it0 = it;
                            }
                            dH = 0;
                        } else {
                            if (uw(i,j) == 0) {
                                dH = 0;
                            } else {
                                float Fr2 = pow(uw(i,j),2)/GRAVITY/h(i,j);
                                if (Fr2 < pow(0.7,2)) {
                                    dH = Cf/GRAVITY/h(i,j)*uw(i,j)*(ux(i,j)*(ip-i)*dx+uy(i,j)*(jp-j)*dx);
                                    dH = Cf*Fr2*(ux(i,j)*(ip-i)*dx+uy(i,j)*(jp-j)*dx)/uw(i,j); // TODO: Why re-assign???
                                    dH = dH+(eta(ip,jp)-eta(i,j));
                                    dH = dH/(1-Fr2);
                                    dH = dH+eta(i,j)-eta(ip,jp);
                                } else {
                                    dH = Cf*Fr2*dx*(ux(i,j)*(ip-i)+uy(i,j)*(jp-j))/uw(i,j);
                                }
                            }
                        }

                        Hnew(i,j) = Hnew(ip,jp)+dH;
                        sfc_visit(i,j) = sfc_visit(i,j)+1;
                        sfc_sum(i,j) = sfc_sum(i,j)+Hnew(i,j);
                    }
                }
            }
        }

        // update flow field
        for (int i=1; i<L; i++) {
            for (int j=1; j<W; j++) {
                float dloc = sqrt(pow(qxn(i,j),2)+pow(qyn(i,j),2));
                if (dloc > 0) {
                    qxn(i,j) = qwn(i,j)*qxn(i,j)/dloc;
                    qyn(i,j) = qwn(i,j)*qyn(i,j)/dloc;
                } else {
                    // aaa = qwn(i,j)
                }
            }
        }
        if (iter > 1) {
            qx = qxn*omega_flow+qx*(1-omega_flow);
            qy = qyn*omega_flow+qy*(1-omega_flow);
        } else {
            qx = qxn;
            qy = qyn;
        }
        qw = (SQUARE(qx)+SQUARE(qy)).cwiseSqrt();

        // start the update of free surface
        for (int i=0; i<L; i++) { // take the average of "sampling" paths
            for (int j=0; j<W; j++) {
                if (sfc_visit(i,j) > 0) {
                    Hnew(i,j) = sfc_sum(i,j)/sfc_visit(i,j);
                }
            }
        }
        Htemp = Hnew; // smoother is applied to newly calculated free surface Hnew
        for (int itsmooth=1; itsmooth<Nsmooth; itsmooth++) {
            Hsmth = Htemp;
            for (int i=1; i<L; i++) {
                for (int j=1; j<W; j++) {
                    if (boundflag(i,j) != 1 && wet_flag(i,j) == 1) {
                        float sumH = 0;
                        int nbcount = 0;
                        vector<int> dxn = nbr(i,j);
                        for (int k=0; k<dxn.size(); k++) {
                            int inbr = i+dxn_iwalk(dxn[k]);
                            int jnbr = j+dxn_jwalk(dxn[k]);
                            if (boundflag(inbr,jnbr) == 0) {
                                sumH += Hsmth(inbr,jnbr);
                                nbcount++;
                            }
                        }
                        if (nbcount == 0) {
                            printf("nbcount is zero @ (%d, %d)\n",i,j);
                        } else {
                            Htemp(i,j) = Csmooth*Hsmth(i,j)+(1-Csmooth)*sumH/nbcount;
                        }
                    }
                }
            }
        }
        Hsmth = Htemp;
        // correction dry/wet
        for (int i=1; i<L; i++) {
            for (int j=1; j<W; j++) {
                if (wet_flag(i,j) == 0) { // locate dry nodes
                    vector<int> dxn = nbr(i,j);
                    for (int k=0; k<dxn.size(); k++) {
                        int inbr = i+dxn_iwalk(dxn[k]);
                        int jnbr = j+dxn_jwalk(dxn[k]);
                        if (wet_flag(inbr,jnbr) == 1 && H(inbr,jnbr)>eta(i,j)) {
                            H(i,j) = H(inbr,jnbr);
                        }
                    }
                }
            }
        }

        // under-relaxation update of the surface
        if (iter > 1) {
            H = (1-omega_sfc)*H+omega_sfc*Hsmth;
        }

        // update flow depth and velocity
        h = (H-eta).cwiseMax(MatrixXf::Zero(L,W)); 
        for (int i=1; i<L; i++) {
            for (int j=1; j<W; j++) {
                if (h(i,j) > h_dry && qw(i,j) > 0) {
                    uw(i,j) = min(u_max,qw(i,j)/h(i,j));
                    ux(i,j) = uw(i,j)*qx(i,j)/qw(i,j);
                    uy(i,j) = uw(i,j)*qy(i,j)/qw(i,j);
                } else {
                    ux(i,j) = 0;
                    uy(i,j) = 0;
                    uw(i,j) = 0;
                }
            }
        }

        // figure(1)

        // subplot(2,1,1)
        // imagesc(H,[0,1])
        // %     title('water surface elevation')
        // axis equal
        // colorbar
        // axis([0,W,0,L])
    
        // subplot(2,1,2)
        // imagesc(qw,[0,5])
        // axis equal
        // %     title('water discharge')
        // colorbar
        // axis([0,W,0,L])
    
        // pause(0.01)

        // if mod(iter,10) == 0
        //     handl = figure(1);
        //     saveas(handl,sprintf('iter = %d',iter),'png')
        // end

        if (iter > maxiter-100) {
            uw_avg = uw_avg+uw/100;
            ux_avg = ux_avg+ux/100;
            uy_avg = uy_avg+uy/100;
            sfc_avg = sfc_avg+H/100;
        }
    
        for (int k=0; k<pxy_start.rows(); k++) {
            P_start(k) = h(pxy_start(k,0),pxy_start(k,1));
        }
        P_start = P_start/P_start.sum();
    }
    cout << double(clock() - startTime) / (double)CLOCKS_PER_SEC<< " seconds." << endl;
}

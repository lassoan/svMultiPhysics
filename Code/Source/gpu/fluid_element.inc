__device__ void d_fluid_3d_c(const FluidParams& prm, const int vmsFlag, const int eNoNw, const int eNoNq, const double w,
    const double Kxi_[3][3], const double* Nw_, const double* Nq_, const double Nwx_[3][4],
    const double Nqx_[3][4], const double Nwxx_[6][4], const double al_[4][4], const double yl_[4][4],
    const double bfl_[3][4], double lR_[4][4], double lK_[16][4][4], double brinkman_inverse_permeability,
    const double urisFactorTotal, const double* urisValveVelTermTotal)
{
      
  

  const double dt = prm.dt;

  const double ctM  = 1.0;
  const double ctC  = 36.0;

  double rho = prm.rho;
  double f[3];
  f[0] = prm.f[0];
  f[1] = prm.f[1];
  f[2] = prm.f[2];

  double T1 = prm.af * prm.gam * dt;
  double amd = prm.am / T1;
  double wl = w*T1;
  double wr = w*rho;

  // Note that indices are not selected based on the equation because
  // fluid equation always come first
  // Velocity and its gradients, inertia (acceleration & body force)
  //
  double ud[3] = {-f[0], -f[1], -f[2]};
  double u[3] = {};
  double ux[3][3] = {};
  double uxx[3][3][3] = {};

  for (int a = 0; a < eNoNw; a++) {
    ud[0] = ud[0] + Nw(a)*(al(0,a)-bfl(0,a));
    ud[1] = ud[1] + Nw(a)*(al(1,a)-bfl(1,a));
    ud[2] = ud[2] + Nw(a)*(al(2,a)-bfl(2,a));

    u[0] = u[0] + Nw(a)*yl(0,a);
    u[1] = u[1] + Nw(a)*yl(1,a);
    u[2] = u[2] + Nw(a)*yl(2,a);

    ux[0][0] = ux[0][0] + Nwx(0,a)*yl(0,a);
    ux[1][0] = ux[1][0] + Nwx(1,a)*yl(0,a);
    ux[2][0] = ux[2][0] + Nwx(2,a)*yl(0,a);
    ux[0][1] = ux[0][1] + Nwx(0,a)*yl(1,a);
    ux[1][1] = ux[1][1] + Nwx(1,a)*yl(1,a);
    ux[2][1] = ux[2][1] + Nwx(2,a)*yl(1,a);
    ux[0][2] = ux[0][2] + Nwx(0,a)*yl(2,a);
    ux[1][2] = ux[1][2] + Nwx(1,a)*yl(2,a);
    ux[2][2] = ux[2][2] + Nwx(2,a)*yl(2,a);

    uxx[0][0][0] += Nwxx(0,a)*yl(0,a);
    uxx[1][0][1] += Nwxx(1,a)*yl(0,a);
    uxx[2][0][2] += Nwxx(2,a)*yl(0,a);
    uxx[1][0][0] += Nwxx(3,a)*yl(0,a);
    uxx[2][0][1] += Nwxx(4,a)*yl(0,a);
    uxx[0][0][2] += Nwxx(5,a)*yl(0,a);

    uxx[0][1][0] += Nwxx(0,a)*yl(1,a);
    uxx[1][1][1] += Nwxx(1,a)*yl(1,a);
    uxx[2][1][2] += Nwxx(2,a)*yl(1,a);
    uxx[1][1][0] += Nwxx(3,a)*yl(1,a);
    uxx[2][1][1] += Nwxx(4,a)*yl(1,a);
    uxx[0][1][2] += Nwxx(5,a)*yl(1,a);

    uxx[0][2][0] += Nwxx(0,a)*yl(2,a);
    uxx[1][2][1] += Nwxx(1,a)*yl(2,a);
    uxx[2][2][2] += Nwxx(2,a)*yl(2,a);
    uxx[1][2][0] += Nwxx(3,a)*yl(2,a);
    uxx[2][2][1] += Nwxx(4,a)*yl(2,a);
    uxx[0][2][2] += Nwxx(5,a)*yl(2,a);
  }

  double divU = ux[0][0] + ux[1][1] + ux[2][2];

  uxx[0][0][1] = uxx[1][0][0];
  uxx[1][0][2] = uxx[2][0][1];
  uxx[2][0][0] = uxx[0][0][2];

  uxx[0][1][1] = uxx[1][1][0];
  uxx[1][1][2] = uxx[2][1][1];
  uxx[2][1][0] = uxx[0][1][2];

  uxx[0][2][1] = uxx[1][2][0];
  uxx[1][2][2] = uxx[2][2][1];
  uxx[2][2][0] = uxx[0][2][2];

  double d2u2[3] = {};
  d2u2[0] = uxx[0][0][0] + uxx[1][0][1] + uxx[2][0][2];
  d2u2[1] = uxx[0][1][0] + uxx[1][1][1] + uxx[2][1][2];
  d2u2[2] = uxx[0][2][0] + uxx[1][2][1] + uxx[2][2][2];

  // Pressure and its gradient
  //
  double px[3] = {};

  for (int a = 0; a < eNoNq; a++) {
    px[0] = px[0] + Nqx(0,a)*yl(3,a);
    px[1] = px[1] + Nqx(1,a)*yl(3,a);
    px[2] = px[2] + Nqx(2,a)*yl(3,a);
  }

  // Update convection velocity relative to mesh velocity
  //
  if (false) {
    for (int a = 0; a < eNoNw; a++) {
      u[0] = u[0] - Nw(a)*yl(4,a);
      u[1] = u[1] - Nw(a)*yl(5,a);
      u[2] = u[2] - Nw(a)*yl(6,a);
    }
  }

  // Strain rate tensor 2*e_ij := (u_i,j + u_j,i)
  //
  double es[3][3] = {};
  es[0][0] = ux[0][0] + ux[0][0];
  es[1][1] = ux[1][1] + ux[1][1];
  es[2][2] = ux[2][2] + ux[2][2];
  es[1][0] = ux[1][0] + ux[0][1];
  es[2][1] = ux[2][1] + ux[1][2];
  es[0][2] = ux[0][2] + ux[2][0];
  es[0][1] = es[1][0];
  es[1][2] = es[2][1];
  es[2][0] = es[0][2];

  double esNx[3][maxNoN];

  for (int a = 0; a < eNoNw; a++) {
    esNx[0][a] = es[0][0]*Nwx(0,a) + es[1][0]*Nwx(1,a) + es[2][0]*Nwx(2,a);
    esNx[1][a] = es[0][1]*Nwx(0,a) + es[1][1]*Nwx(1,a) + es[2][1]*Nwx(2,a);
    esNx[2][a] = es[0][2]*Nwx(0,a) + es[1][2]*Nwx(1,a) + es[2][2]*Nwx(2,a);
  }

  double es_x[3][3][3] = {};

  for (int k = 0; k < 3; k++) { 
     es_x[0][0][k] = uxx[0][0][k] + uxx[0][0][k];
     es_x[1][1][k] = uxx[1][1][k] + uxx[1][1][k];
     es_x[2][2][k] = uxx[2][2][k] + uxx[2][2][k];
     es_x[1][0][k] = uxx[1][0][k] + uxx[0][1][k];
     es_x[2][1][k] = uxx[2][1][k] + uxx[1][2][k];
     es_x[0][2][k] = uxx[0][2][k] + uxx[2][0][k];

     es_x[0][1][k] = es_x[1][0][k];
     es_x[1][2][k] = es_x[2][1][k];
     es_x[2][0][k] = es_x[0][2][k];
  }

  double mu_x[3];

  mu_x[0] = (es_x[0][0][0]*es[0][0] + es_x[1][1][0]*es[1][1]
          +  es_x[2][2][0]*es[2][2])*0.5
          +  es_x[1][0][0]*es[1][0] + es_x[2][1][0]*es[2][1]
          +  es_x[0][2][0]*es[0][2];

  mu_x[1] = (es_x[0][0][1]*es[0][0] + es_x[1][1][1]*es[1][1]
          +  es_x[2][2][1]*es[2][2])*0.5
          +  es_x[1][0][1]*es[1][0] + es_x[2][1][1]*es[2][1]
          +  es_x[0][2][1]*es[0][2];

  mu_x[2] = (es_x[0][0][2]*es[0][0] + es_x[1][1][2]*es[1][1]
          +  es_x[2][2][2]*es[2][2])*0.5
          +  es_x[1][0][2]*es[1][0] + es_x[2][1][2]*es[2][1]
          +  es_x[0][2][2]*es[0][2];

  // Shear-rate := (2*e_ij*e_ij)^.5
  double gam = es[0][0]*es[0][0] + es[1][0]*es[1][0] + es[2][0]*es[2][0]
             + es[0][1]*es[0][1] + es[1][1]*es[1][1] + es[2][1]*es[2][1]
             + es[0][2]*es[0][2] + es[1][2]*es[1][2] + es[2][2]*es[2][2];
  gam = sqrt(0.5*gam);

  // Compute viscosity based on shear-rate and chosen viscosity model
  // The returned mu_g := (d\mu / d\gamma)
  double mu, mu_s, mu_g;
  mu = prm.mu; mu_s = mu; mu_g = 0.0;

  if (is_zero(gam)) {
     mu_g = 0.0;
  } else {
     mu_g = mu_g / gam;
  }

  for (int i = 0; i < 3; i++) {
    mu_x[i] = mu_g * mu_x[i];
  }

  // Stabilization parameters
  //
  double up[3] = {};
  double updu[3][3][maxNoN] = {};
  double tauM = 0.0;

  if (vmsFlag) {
    // Stabilization parameters
    double kT = 4.0 * pow(ctM/dt,2.0);
    
    // If we consider the NSB model, we need to add an extra term inside the computation for the stab parameter 
    kT = kT + pow(brinkman_inverse_permeability*mu/rho, 2.0);
    
    // In case of unfitted RIS, compute the delta function at the quad point,
    // add the additional value to the stabilization param 
    kT = kT + pow(urisFactorTotal, 2.0);

    double kU = u[0]*u[0]*Kxi(0,0) + u[1]*u[0]*Kxi(1,0) + u[2]*u[0]*Kxi(2,0)
              + u[0]*u[1]*Kxi(0,1) + u[1]*u[1]*Kxi(1,1) + u[2]*u[1]*Kxi(2,1)
              + u[0]*u[2]*Kxi(0,2) + u[1]*u[2]*Kxi(1,2) + u[2]*u[2]*Kxi(2,2);

    double kS = Kxi(0,0)*Kxi(0,0) + Kxi(1,0)*Kxi(1,0) + Kxi(2,0)*Kxi(2,0)
              + Kxi(0,1)*Kxi(0,1) + Kxi(1,1)*Kxi(1,1) + Kxi(2,1)*Kxi(2,1)
              + Kxi(0,2)*Kxi(0,2) + Kxi(1,2)*Kxi(1,2) + Kxi(2,2)*Kxi(2,2);

    kS = ctC * kS * pow(mu/rho,2.0);
    tauM = 1.0 / (rho * sqrt( kT + kU + kS ));

    double rV[3];
    rV[0] = ud[0] + u[0]*ux[0][0] + u[1]*ux[1][0] + u[2]*ux[2][0];
    rV[1] = ud[1] + u[0]*ux[0][1] + u[1]*ux[1][1] + u[2]*ux[2][1];
    rV[2] = ud[2] + u[0]*ux[0][2] + u[1]*ux[1][2] + u[2]*ux[2][2];

    double rS[3];
    rS[0] = mu_x[0]*es[0][0] + mu_x[1]*es[1][0] + mu_x[2]*es[2][0] + mu*d2u2[0];
    rS[1] = mu_x[0]*es[0][1] + mu_x[1]*es[1][1] + mu_x[2]*es[2][1] + mu*d2u2[1];
    rS[2] = mu_x[0]*es[0][2] + mu_x[1]*es[1][2] + mu_x[2]*es[2][2] + mu*d2u2[2];

    // up[0] = -tauM*(rho*rV[0] + px[0] - rS[0] + mu*brinkman_inverse_permeability*u[0]);
    // up[1] = -tauM*(rho*rV[1] + px[1] - rS[1] + mu*brinkman_inverse_permeability*u[1]);
    // up[2] = -tauM*(rho*rV[2] + px[2] - rS[2] + mu*brinkman_inverse_permeability*u[2]);

    up[0] = -tauM*(rho*rV[0] + px[0] - rS[0] + mu*brinkman_inverse_permeability*u[0]
                   + urisFactorTotal*u[0] - urisValveVelTermTotal[0]);
    up[1] = -tauM*(rho*rV[1] + px[1] - rS[1] + mu*brinkman_inverse_permeability*u[1]
                   + urisFactorTotal*u[1] - urisValveVelTermTotal[1]);
    up[2] = -tauM*(rho*rV[2] + px[2] - rS[2] + mu*brinkman_inverse_permeability*u[2]
                   + urisFactorTotal*u[2] - urisValveVelTermTotal[2]);

    for (int a = 0; a < eNoNw; a++) {
      double uNx = u[0]*Nwx(0,a) + u[1]*Nwx(1,a) + u[2]*Nwx(2,a);
      // T1 = -rho*uNx + mu*(Nwxx(0,a) + Nwxx(1,a) + Nwxx(2,a)) + mu_x[0]*Nwx(0,a) + mu_x[1]*Nwx(1,a) + mu_x[2]*Nwx(2,a) - mu*brinkman_inverse_permeability*Nw(a);

      T1 = -rho*uNx + mu*(Nwxx(0,a) + Nwxx(1,a) + Nwxx(2,a)) 
           + mu_x[0]*Nwx(0,a) + mu_x[1]*Nwx(1,a) + mu_x[2]*Nwx(2,a) 
           - mu*brinkman_inverse_permeability*Nw(a)
           - urisFactorTotal*Nw(a);

      updu[0][0][a] = mu_x[0]*Nwx(0,a) + d2u2[0]*mu_g*esNx[0][a] + T1;
      updu[1][0][a] = mu_x[1]*Nwx(0,a) + d2u2[1]*mu_g*esNx[0][a];
      updu[2][0][a] = mu_x[2]*Nwx(0,a) + d2u2[2]*mu_g*esNx[0][a];
  
      updu[0][1][a] = mu_x[0]*Nwx(1,a) + d2u2[0]*mu_g*esNx[1][a];
      updu[1][1][a] = mu_x[1]*Nwx(1,a) + d2u2[1]*mu_g*esNx[1][a] + T1;
      updu[2][1][a] = mu_x[2]*Nwx(1,a) + d2u2[2]*mu_g*esNx[1][a];
  
      updu[0][2][a] = mu_x[0]*Nwx(2,a) + d2u2[0]*mu_g*esNx[2][a];
      updu[1][2][a] = mu_x[1]*Nwx(2,a) + d2u2[1]*mu_g*esNx[2][a];
      updu[2][2][a] = mu_x[2]*Nwx(2,a) + d2u2[2]*mu_g*esNx[2][a] + T1;
    }

  } else {
    tauM = 0.0;
    memset(up, 0, sizeof up);
    memset(updu, 0, sizeof updu);
  }

  //  Local residual
  //
  for (int a = 0; a < eNoNq; a++) {
    double upNx = up[0]*Nqx(0,a) + up[1]*Nqx(1,a) + up[2]*Nqx(2,a);
    lR(3,a) = lR(3,a) + w*(Nq(a)*divU - upNx);
  }

  // Tangent (stiffness) matrices
  //
  for (int b = 0; b < eNoNw; b++) {
    T1 = rho*amd*Nw(b);

    for (int a = 0; a < eNoNq; a++) {
      // dRc_a/dU_b1
      double T2 = Nqx(0,a)*(updu[0][0][b] - T1) + Nqx(1,a)*updu[0][1][b] + Nqx(2,a)*updu[0][2][b];
      lK(12,a,b) = lK(12,a,b) + wl*(Nq(a)*Nwx(0,b) - tauM*T2);

      // dRc_a/dU_b2
      T2 = Nqx(0,a)*updu[1][0][b] + Nqx(1,a)*(updu[1][1][b] - T1) + Nqx(2,a)*updu[1][2][b];
      lK(13,a,b) = lK(13,a,b) + wl*(Nq(a)*Nwx(1,b) - tauM*T2);

      // dRc_a/dU_b3
      T2 = Nqx(0,a)*updu[2][0][b] + Nqx(1,a)*updu[2][1][b] + Nqx(2,a)*(updu[2][2][b] - T1);
      lK(14,a,b) = lK(14,a,b) + wl*(Nq(a)*Nwx(2,b) - tauM*T2);
    }
  }

  if (vmsFlag) {
    for (int b = 0; b < eNoNq; b++) {
      for (int a = 0; a < eNoNq; a++) {
        // dC/dP
        double NxNx = Nqx(0,a)*Nqx(0,b) + Nqx(1,a)*Nqx(1,b) + Nqx(2,a)*Nqx(2,b);
        lK(15,a,b) = lK(15,a,b) + wl*tauM*NxNx;
      }
    }
  }
}

__device__ void d_fluid_3d_m(const FluidParams& prm, const int vmsFlag, const int eNoNw, const int eNoNq, const double w,
    const double Kxi_[3][3], const double* Nw_, const double* Nq_, const double Nwx_[3][4],
    const double Nqx_[3][4], const double Nwxx_[6][4], const double al_[4][4], const double yl_[4][4],
    const double bfl_[3][4], double lR_[4][4], double lK_[16][4][4], double brinkman_inverse_permeability,
    const double urisFactorTotal, const double* urisValveVelTermTotal)
{
    
  

  const double dt = prm.dt;

  double ctM  = 1.0;
  double ctC  = 36.0;

  double rho = prm.rho;
  double f[3];
  f[0] = prm.f[0];
  f[1] = prm.f[1];
  f[2] = prm.f[2];

  double T1 = prm.af * prm.gam * dt;
  double amd = prm.am / T1;
  double wl = w*T1;
  double wr = w*rho;

  
  // Note that indices are not selected based on the equation because
  // fluid equation always come first
  // Velocity and its gradients, inertia (acceleration & body force)
  //
  double ud[3] = {-f[0], -f[1], -f[2]};
  //ud  = -f
  double u[3] = {};
  double ux[3][3] = {};
  double uxx[3][3][3] = {};

  for (int a = 0; a < eNoNw; a++) {
    ud[0] = ud[0] + Nw(a)*(al(0,a)-bfl(0,a));
    ud[1] = ud[1] + Nw(a)*(al(1,a)-bfl(1,a));
    ud[2] = ud[2] + Nw(a)*(al(2,a)-bfl(2,a));

    u[0] = u[0] + Nw(a)*yl(0,a);
    u[1] = u[1] + Nw(a)*yl(1,a);
    u[2] = u[2] + Nw(a)*yl(2,a);
    
    ux[0][0] += Nwx(0,a)*yl(0,a);
    ux[1][0] += Nwx(1,a)*yl(0,a);
    ux[2][0] += Nwx(2,a)*yl(0,a);
    ux[0][1] += Nwx(0,a)*yl(1,a);
    ux[1][1] += Nwx(1,a)*yl(1,a);
    ux[2][1] += Nwx(2,a)*yl(1,a);
    ux[0][2] += Nwx(0,a)*yl(2,a);
    ux[1][2] += Nwx(1,a)*yl(2,a);
    ux[2][2] += Nwx(2,a)*yl(2,a);

    uxx[0][0][0] += Nwxx(0,a)*yl(0,a);
    uxx[1][0][1] += Nwxx(1,a)*yl(0,a);
    uxx[2][0][2] += Nwxx(2,a)*yl(0,a);
    uxx[1][0][0] += Nwxx(3,a)*yl(0,a);
    uxx[2][0][1] += Nwxx(4,a)*yl(0,a);
    uxx[0][0][2] += Nwxx(5,a)*yl(0,a);

    uxx[0][1][0] += Nwxx(0,a)*yl(1,a);
    uxx[1][1][1] += Nwxx(1,a)*yl(1,a);
    uxx[2][1][2] += Nwxx(2,a)*yl(1,a);
    uxx[1][1][0] += Nwxx(3,a)*yl(1,a);
    uxx[2][1][1] += Nwxx(4,a)*yl(1,a);
    uxx[0][1][2] += Nwxx(5,a)*yl(1,a);

    uxx[0][2][0] += Nwxx(0,a)*yl(2,a);
    uxx[1][2][1] += Nwxx(1,a)*yl(2,a);
    uxx[2][2][2] += Nwxx(2,a)*yl(2,a);
    uxx[1][2][0] += Nwxx(3,a)*yl(2,a);
    uxx[2][2][1] += Nwxx(4,a)*yl(2,a);
    uxx[0][2][2] += Nwxx(5,a)*yl(2,a);
  }

  double divU = ux[0][0] + ux[1][1] + ux[2][2];
  
  uxx[0][0][1] = uxx[1][0][0];
  uxx[1][0][2] = uxx[2][0][1];
  uxx[2][0][0] = uxx[0][0][2];

  uxx[0][1][1] = uxx[1][1][0];
  uxx[1][1][2] = uxx[2][1][1];
  uxx[2][1][0] = uxx[0][1][2];

  uxx[0][2][1] = uxx[1][2][0];
  uxx[1][2][2] = uxx[2][2][1];
  uxx[2][2][0] = uxx[0][2][2];

  double d2u2[3] = {0.0};
  d2u2[0] = uxx[0][0][0] + uxx[1][0][1] + uxx[2][0][2];
  d2u2[1] = uxx[0][1][0] + uxx[1][1][1] + uxx[2][1][2];
  d2u2[2] = uxx[0][2][0] + uxx[1][2][1] + uxx[2][2][2];

  // Pressure and its gradient
  //
  double p = 0.0;
  double px[3] = {};

  for (int a = 0; a < eNoNq; a++) {
    p  = p + Nq(a)*yl(3,a);
    px[0] = px[0] + Nqx(0,a)*yl(3,a);
    px[1] = px[1] + Nqx(1,a)*yl(3,a);
    px[2] = px[2] + Nqx(2,a)*yl(3,a);
  }
  
  // Update convection velocity relative to mesh velocity
  //
  if (false) {
    for (int a = 0; a < eNoNw; a++) {
      u[0] = u[0] - Nw(a)*yl(4,a);
      u[1] = u[1] - Nw(a)*yl(5,a);
      u[2] = u[2] - Nw(a)*yl(6,a);
    }
  }

  // Strain rate tensor 2*e_ij := (u_i,j + u_j,i)
  //
  double es[3][3] = {};
  es[0][0] = ux[0][0] + ux[0][0];
  es[1][1] = ux[1][1] + ux[1][1];
  es[2][2] = ux[2][2] + ux[2][2];
  es[1][0] = ux[1][0] + ux[0][1];
  es[2][1] = ux[2][1] + ux[1][2];
  es[0][2] = ux[0][2] + ux[2][0];
  es[0][1] = es[1][0];
  es[1][2] = es[2][1];
  es[2][0] = es[0][2];

  double esNx[3][maxNoN];

  for (int a = 0; a < eNoNw; a++) {
    esNx[0][a] = es[0][0]*Nwx(0,a) + es[1][0]*Nwx(1,a) + es[2][0]*Nwx(2,a);
    esNx[1][a] = es[0][1]*Nwx(0,a) + es[1][1]*Nwx(1,a) + es[2][1]*Nwx(2,a);
    esNx[2][a] = es[0][2]*Nwx(0,a) + es[1][2]*Nwx(1,a) + es[2][2]*Nwx(2,a);
  }

  double es_x[3][3][3];

  for (int k = 0; k < 3; k++) { 
     es_x[0][0][k] = uxx[0][0][k] + uxx[0][0][k];
     es_x[1][1][k] = uxx[1][1][k] + uxx[1][1][k];
     es_x[2][2][k] = uxx[2][2][k] + uxx[2][2][k];
     es_x[1][0][k] = uxx[1][0][k] + uxx[0][1][k];
     es_x[2][1][k] = uxx[2][1][k] + uxx[1][2][k];
     es_x[0][2][k] = uxx[0][2][k] + uxx[2][0][k];

     es_x[0][1][k] = es_x[1][0][k];
     es_x[1][2][k] = es_x[2][1][k];
     es_x[2][0][k] = es_x[0][2][k];
  }

  double mu_x[3] = {0.0};

  mu_x[0] = (es_x[0][0][0]*es[0][0] + es_x[1][1][0]*es[1][1]
          +  es_x[2][2][0]*es[2][2])*0.5
          +  es_x[1][0][0]*es[1][0] + es_x[2][1][0]*es[2][1]
          +  es_x[0][2][0]*es[0][2];

  mu_x[1] = (es_x[0][0][1]*es[0][0] + es_x[1][1][1]*es[1][1]
          +  es_x[2][2][1]*es[2][2])*0.5
          +  es_x[1][0][1]*es[1][0] + es_x[2][1][1]*es[2][1]
          +  es_x[0][2][1]*es[0][2];

  mu_x[2] = (es_x[0][0][2]*es[0][0] + es_x[1][1][2]*es[1][1]
          +  es_x[2][2][2]*es[2][2])*0.5
          +  es_x[1][0][2]*es[1][0] + es_x[2][1][2]*es[2][1]
          +  es_x[0][2][2]*es[0][2];

  
  // Shear-rate := (2*e_ij*e_ij)^.5
  //
  //dmsg << "Compute shear rate ... ";
  double gam = es[0][0]*es[0][0] + es[1][0]*es[1][0] + es[2][0]*es[2][0]
             + es[0][1]*es[0][1] + es[1][1]*es[1][1] + es[2][1]*es[2][1]
             + es[0][2]*es[0][2] + es[1][2]*es[1][2] + es[2][2]*es[2][2];
  gam = sqrt(0.5*gam);
  
  // Compute viscosity based on shear-rate and chosen viscosity model
  // The returned mu_g := (d\mu / d\gamma)
  //
  double mu, mu_s, mu_g;
  mu = prm.mu; mu_s = mu; mu_g = 0.0;

  if (is_zero(gam)) {
     mu_g = 0.0;
  } else {
     mu_g = mu_g / gam;
  }
  for (int i = 0; i < 3; i++) mu_x[i] = mu_g * mu_x[i];
  //mu_x(:) = mu_g * mu_x(:)

  // Stabilization parameters
  //
  double kT = 4.0 * pow(ctM/dt,2.0);
  
  // If we consider the NSB model, we need to add an extra term inside the computation for the stab parameter 
  kT = kT + pow(brinkman_inverse_permeability*mu/rho, 2.0);

  // In case of unfitted RIS, compute the delta function at the quad point,
  // add the additional value to the stabilization param 
  kT = kT + pow(urisFactorTotal, 2.0);

  double kU = u[0]*u[0]*Kxi(0,0) + u[1]*u[0]*Kxi(1,0) + u[2]*u[0]*Kxi(2,0)
            + u[0]*u[1]*Kxi(0,1) + u[1]*u[1]*Kxi(1,1) + u[2]*u[1]*Kxi(2,1)
            + u[0]*u[2]*Kxi(0,2) + u[1]*u[2]*Kxi(1,2) + u[2]*u[2]*Kxi(2,2);

  double kS = Kxi(0,0)*Kxi(0,0) + Kxi(1,0)*Kxi(1,0) + Kxi(2,0)*Kxi(2,0)
            + Kxi(0,1)*Kxi(0,1) + Kxi(1,1)*Kxi(1,1) + Kxi(2,1)*Kxi(2,1)
            + Kxi(0,2)*Kxi(0,2) + Kxi(1,2)*Kxi(1,2) + Kxi(2,2)*Kxi(2,2);
  kS = ctC * kS * pow(mu/rho,2.0);
  double tauM = 1.0 / (rho * sqrt( kT + kU + kS ));

  
  double rV[3] = {};
  rV[0] = ud[0] + u[0]*ux[0][0] + u[1]*ux[1][0] + u[2]*ux[2][0];
  rV[1] = ud[1] + u[0]*ux[0][1] + u[1]*ux[1][1] + u[2]*ux[2][1];
  rV[2] = ud[2] + u[0]*ux[0][2] + u[1]*ux[1][2] + u[2]*ux[2][2];

  double rS[3] = {};
  rS[0] = mu_x[0]*es[0][0] + mu_x[1]*es[1][0] + mu_x[2]*es[2][0] + mu*d2u2[0];
  rS[1] = mu_x[0]*es[0][1] + mu_x[1]*es[1][1] + mu_x[2]*es[2][1] + mu*d2u2[1];
  rS[2] = mu_x[0]*es[0][2] + mu_x[1]*es[1][2] + mu_x[2]*es[2][2] + mu*d2u2[2];

  double up[3] = {};
  // up[0] = -tauM*(rho*rV[0] + px[0] - rS[0] + mu*brinkman_inverse_permeability * u[0]);
  // up[1] = -tauM*(rho*rV[1] + px[1] - rS[1] + mu*brinkman_inverse_permeability * u[1]);
  // up[2] = -tauM*(rho*rV[2] + px[2] - rS[2] + mu*brinkman_inverse_permeability * u[2]);

  up[0] = -tauM*(rho*rV[0] + px[0] - rS[0] + mu*brinkman_inverse_permeability * u[0]
                 + urisFactorTotal * u[0] - urisValveVelTermTotal[0]);
  up[1] = -tauM*(rho*rV[1] + px[1] - rS[1] + mu*brinkman_inverse_permeability * u[1]
                 + urisFactorTotal * u[1] - urisValveVelTermTotal[1]);
  up[2] = -tauM*(rho*rV[2] + px[2] - rS[2] + mu*brinkman_inverse_permeability * u[2]
                 + urisFactorTotal * u[2] - urisValveVelTermTotal[2]);

  double tauC, tauB, pa;
  double eps = DBL_EPSILON;
  double ua[3] = {};

  if (vmsFlag) {
    tauC = 1.0 / (tauM * (Kxi(0,0) + Kxi(1,1) + Kxi(2,2)));
    tauB = up[0]*up[0]*Kxi(0,0) + up[1]*up[0]*Kxi(1,0)
         + up[2]*up[0]*Kxi(2,0) + up[0]*up[1]*Kxi(0,1)
         + up[1]*up[1]*Kxi(1,1) + up[2]*up[1]*Kxi(2,1)
         + up[0]*up[2]*Kxi(0,2) + up[1]*up[2]*Kxi(1,2)
         + up[2]*up[2]*Kxi(2,2);

    if (is_zero(tauB)) {
      tauB = eps;
    }
    tauB = rho / sqrt(tauB);

    ua[0] = u[0] + up[0];
    ua[1] = u[1] + up[1];
    ua[2] = u[2] + up[2];
    pa = p - tauC*divU;

  } else {
    tauC = 0.0;
    tauB = 0.0;
    for (int i = 0; i < 3; i++) {
      ua[i] = u[i];
    }
    pa = p;
  }

  rV[0] = tauB*(up[0]*ux[0][0] + up[1]*ux[1][0] + up[2]*ux[2][0]);
  rV[1] = tauB*(up[0]*ux[0][1] + up[1]*ux[1][1] + up[2]*ux[2][1]);
  rV[2] = tauB*(up[0]*ux[0][2] + up[1]*ux[1][2] + up[2]*ux[2][2]);

  double rM[3][3];
  rM[0][0] = mu*es[0][0] - rho*up[0]*ua[0] + rV[0]*up[0] - pa;
  rM[1][0] = mu*es[1][0] - rho*up[0]*ua[1] + rV[0]*up[1];
  rM[2][0] = mu*es[2][0] - rho*up[0]*ua[2] + rV[0]*up[2];

  rM[0][1] = mu*es[0][1] - rho*up[1]*ua[0] + rV[1]*up[0];
  rM[1][1] = mu*es[1][1] - rho*up[1]*ua[1] + rV[1]*up[1] - pa;
  rM[2][1] = mu*es[2][1] - rho*up[1]*ua[2] + rV[1]*up[2];

  rM[0][2] = mu*es[0][2] - rho*up[2]*ua[0] + rV[2]*up[0];
  rM[1][2] = mu*es[1][2] - rho*up[2]*ua[1] + rV[2]*up[1];
  rM[2][2] = mu*es[2][2] - rho*up[2]*ua[2] + rV[2]*up[2] - pa;

  rV[0] = ud[0] + ua[0]*ux[0][0] + ua[1]*ux[1][0] + ua[2]*ux[2][0];
  rV[1] = ud[1] + ua[0]*ux[0][1] + ua[1]*ux[1][1] + ua[2]*ux[2][1];
  rV[2] = ud[2] + ua[0]*ux[0][2] + ua[1]*ux[1][2] + ua[2]*ux[2][2];

  //  Local residual
  //
  double updu[3][3][maxNoN] = {};
  double uNx[maxNoN] = {};
  double upNx[maxNoN] = {}; 
  double uaNx[maxNoN] = {}; 

  for (int a = 0; a < eNoNw; a++) {
    lR(0,a) = lR(0,a) + wr*Nw(a)*rV[0] + w*(Nwx(0,a)*rM[0][0] + Nwx(1,a)*rM[1][0] + Nwx(2,a)*rM[2][0]);
    lR(1,a) = lR(1,a) + wr*Nw(a)*rV[1] + w*(Nwx(0,a)*rM[0][1] + Nwx(1,a)*rM[1][1] + Nwx(2,a)*rM[2][1]);
    lR(2,a) = lR(2,a) + wr*Nw(a)*rV[2] + w*(Nwx(0,a)*rM[0][2] + Nwx(1,a)*rM[1][2] + Nwx(2,a)*rM[2][2]);

    // Quantities used for stiffness matrix

    uNx[a]  = u[0]*Nwx(0,a)  + u[1]*Nwx(1,a)  + u[2]*Nwx(2,a);
    upNx[a] = up[0]*Nwx(0,a) + up[1]*Nwx(1,a) + up[2]*Nwx(2,a);

    if (vmsFlag) {
       uaNx[a] = uNx[a] + upNx[a];
    } else {
       uaNx[a] = uNx[a];
    }

    // T1 = -rho*uNx[a] + mu*(Nwxx(0,a) + Nwxx(1,a) + Nwxx(2,a)) + mu_x[0]*Nwx(0,a) + mu_x[1]*Nwx(1,a) + mu_x[2]*Nwx(2,a) - mu*brinkman_inverse_permeability*Nw(a);

    T1 = -rho*uNx[a] + mu*(Nwxx(0,a) + Nwxx(1,a) + Nwxx(2,a)) 
         + mu_x[0]*Nwx(0,a) + mu_x[1]*Nwx(1,a) + mu_x[2]*Nwx(2,a) 
         - mu*brinkman_inverse_permeability*Nw(a)
         - urisFactorTotal*Nw(a);

    updu[0][0][a] = mu_x[0]*Nwx(0,a) + d2u2[0]*mu_g*esNx[0][a] + T1;
    updu[1][0][a] = mu_x[1]*Nwx(0,a) + d2u2[1]*mu_g*esNx[0][a];
    updu[2][0][a] = mu_x[2]*Nwx(0,a) + d2u2[2]*mu_g*esNx[0][a];

    updu[0][1][a] = mu_x[0]*Nwx(1,a) + d2u2[0]*mu_g*esNx[1][a];
    updu[1][1][a] = mu_x[1]*Nwx(1,a) + d2u2[1]*mu_g*esNx[1][a] + T1;
    updu[2][1][a] = mu_x[2]*Nwx(1,a) + d2u2[2]*mu_g*esNx[1][a];

    updu[0][2][a] = mu_x[0]*Nwx(2,a) + d2u2[0]*mu_g*esNx[2][a];
    updu[1][2][a] = mu_x[1]*Nwx(2,a) + d2u2[1]*mu_g*esNx[2][a];
    updu[2][2][a] = mu_x[2]*Nwx(2,a) + d2u2[2]*mu_g*esNx[2][a] + T1;
  }

  // Tangent (stiffness) matrices
  //
  for (int b = 0; b < eNoNw; b++) {
    for (int a = 0; a < eNoNw; a++) {
      rM[0][0] = Nwx(0,a)*Nwx(0,b);
      rM[1][0] = Nwx(1,a)*Nwx(0,b);
      rM[2][0] = Nwx(2,a)*Nwx(0,b);
      rM[0][1] = Nwx(0,a)*Nwx(1,b);
      rM[1][1] = Nwx(1,a)*Nwx(1,b);
      rM[2][1] = Nwx(2,a)*Nwx(1,b);
      rM[0][2] = Nwx(0,a)*Nwx(2,b);
      rM[1][2] = Nwx(1,a)*Nwx(2,b);
      rM[2][2] = Nwx(2,a)*Nwx(2,b);

      double NxNx = Nwx(0,a)*Nwx(0,b) + Nwx(1,a)*Nwx(1,b) + Nwx(2,a)*Nwx(2,b);
      T1 = mu*NxNx + rho*amd*Nw(b)*(Nw(a) + rho*tauM*uaNx[a]) + rho*Nw(a)*(uNx[b]+upNx[b]) + tauB*upNx[a]*upNx[b];

      // dRm_a1/du_b1
      double T2 = (mu + tauC)*rM[0][0] + esNx[0][a]*mu_g*esNx[0][b] - rho*tauM*uaNx[a]*updu[0][0][b];
      lK(0,a,b)  = lK(0,a,b)  + wl*(T2 + T1);
      // lK(0,a,b)  = lK(0,a,b)  + mu*brinkman_inverse_permeability*wl*Nw(b)*Nw(a);
      lK(0,a,b)  = lK(0,a,b)  + mu*brinkman_inverse_permeability*wl*Nw(b)*Nw(a)
                              + urisFactorTotal*wl*Nw(b)*Nw(a);

      // dRm_a1/du_b2
      T2 = mu*rM[1][0] + tauC*rM[0][1] + esNx[0][a]*mu_g*esNx[1][b] - rho*tauM*uaNx[a]*updu[1][0][b];
      lK(1,a,b)  = lK(1,a,b)  + wl*(T2);

      // dRm_a1/du_b3
      T2 = mu*rM[2][0] + tauC*rM[0][2] + esNx[0][a]*mu_g*esNx[2][b] - rho*tauM*uaNx[a]*updu[2][0][b];
      lK(2,a,b)  = lK(2,a,b)  + wl*(T2);

      // dRm_a2/du_b1
      T2 = mu*rM[0][1] + tauC*rM[1][0] + esNx[1][a]*mu_g*esNx[0][b] - rho*tauM*uaNx[a]*updu[0][1][b];
      lK(4,a,b) = lK(4,a,b)  + wl*(T2);

      // dRm_a2/du_b2
      T2 = (mu + tauC)*rM[1][1] + esNx[1][a]*mu_g*esNx[1][b] - rho*tauM*uaNx[a]*updu[1][1][b];
      lK(5,a,b)  = lK(5,a,b)  + wl*(T2 + T1);
      // lK(5,a,b)  = lK(5,a,b)  + mu*brinkman_inverse_permeability*wl*Nw(b)*Nw(a);
      lK(5,a,b)  = lK(5,a,b)  + mu*brinkman_inverse_permeability*wl*Nw(b)*Nw(a)
                              + urisFactorTotal*wl*Nw(b)*Nw(a);

      // dRm_a2/du_b3
      T2 = mu*rM[2][1] + tauC*rM[1][2] + esNx[1][a]*mu_g*esNx[2][b] - rho*tauM*uaNx[a]*updu[2][1][b];
      lK(6,a,b)  = lK(6,a,b)  + wl*(T2);

      // dRm_a3/du_b1
      T2 = mu*rM[0][2] + tauC*rM[2][0] + esNx[2][a]*mu_g*esNx[0][b] - rho*tauM*uaNx[a]*updu[0][2][b];
      lK(8,a,b)  = lK(8,a,b)  + wl*(T2);

      // dRm_a3/du_b2
      T2 = mu*rM[1][2] + tauC*rM[2][1] + esNx[2][a]*mu_g*esNx[1][b] - rho*tauM*uaNx[a]*updu[1][2][b];
      lK(9,a,b) = lK(9,a,b) + wl*(T2);

      // dRm_a3/du_b3;
      T2 = (mu + tauC)*rM[2][2] + esNx[2][a]*mu_g*esNx[2][b] - rho*tauM*uaNx[a]*updu[2][2][b];
      lK(10,a,b) = lK(10,a,b) + wl*(T2 + T1);
      // lK(10,a,b) = lK(10,a,b) + mu*brinkman_inverse_permeability*wl*Nw(b)*Nw(a);
      lK(10,a,b) = lK(10,a,b) + mu*brinkman_inverse_permeability*wl*Nw(b)*Nw(a)
                              + urisFactorTotal*wl*Nw(b)*Nw(a);
      //dmsg << "lK(10,a,b): " << lK(10,a,b);
    }
  }

  //exit(0);

  for (int b = 0; b < eNoNq; b++) {
    for (int a = 0; a < eNoNw; a++) {
      T1 = rho*tauM*uaNx[a];

      // dRm_a1/dp_b
      lK(3,a,b)  = lK(3,a,b)  - wl*(Nwx(0,a)*Nq(b) - Nqx(0,b)*T1);

      // dRm_a2/dp_b
      lK(7,a,b)  = lK(7,a,b)  - wl*(Nwx(1,a)*Nq(b) - Nqx(1,b)*T1);

      // dRm_a3/dp_b
      lK(11,a,b) = lK(11,a,b) - wl*(Nwx(2,a)*Nq(b) - Nqx(2,b)*T1);
    }
  }
  
  // Residual contribution Birkman term 
  // Local residue
  for (int a = 0; a < eNoNw; a++) {
      lR(0,a) = lR(0,a) + mu*brinkman_inverse_permeability*w*Nw(a)*(u[0]+up[0])
                        + w*Nw(a)*(urisFactorTotal*u[0] - urisValveVelTermTotal[0]);
      lR(1,a) = lR(1,a) + mu*brinkman_inverse_permeability*w*Nw(a)*(u[1]+up[1])
                        + w*Nw(a)*(urisFactorTotal*u[1] - urisValveVelTermTotal[1]);
      lR(2,a) = lR(2,a) + mu*brinkman_inverse_permeability*w*Nw(a)*(u[2]+up[2])
                        + w*Nw(a)*(urisFactorTotal*u[2] - urisValveVelTermTotal[2]);
  }

}

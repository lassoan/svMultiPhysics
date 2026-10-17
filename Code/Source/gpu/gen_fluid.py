# Generate CUDA device versions of fluid_3d_m and fluid_3d_c from fluid.cpp
import re, sys
src = open(sys.argv[1]).read()
def body(name):
    i = src.index('void ' + name + '(')
    i = src.index('{', i)
    depth = 0
    for j in range(i, len(src)):
        if src[j] == '{': depth += 1
        elif src[j] == '}':
            depth -= 1
            if depth == 0: return src[i+1:j]
out = []
for name in ('fluid_3d_c', 'fluid_3d_m'):
    b = body(name)
    # drop debug blocks
    b = re.sub(r'#define n_debug\w*\n', '', b)
    b = re.sub(r'#ifdef debug\w*\n.*?#endif\n', '', b, flags=re.S)
    b = b.replace('using namespace consts;', '')
    # parameters instead of com_mod
    b = re.sub(r'  int cEq = com_mod.cEq;\n  auto& eq = com_mod.eq\[cEq\];\n  int cDmn = com_mod.cDmn;\n  auto& dmn = eq.dmn\[cDmn\];\n  const double dt = com_mod.dt;\n',
               '  const double dt = prm.dt;\n', b)
    b = b.replace('dmn.prop[PhysicalPropertyType::fluid_density]', 'prm.rho')
    b = b.replace('dmn.prop[PhysicalPropertyType::f_x]', 'prm.f[0]')
    b = b.replace('dmn.prop[PhysicalPropertyType::f_y]', 'prm.f[1]')
    b = b.replace('dmn.prop[PhysicalPropertyType::f_z]', 'prm.f[2]')
    b = b.replace('eq.af', 'prm.af').replace('eq.gam', 'prm.gam').replace('eq.am', 'prm.am')
    b = b.replace('com_mod.mvMsh', 'false')
    b = re.sub(r'(fluid::)?get_viscosity\(com_mod, dmn, gam, mu, mu_s, mu_g\);', 'mu = prm.mu; mu_s = mu; mu_g = 0.0;', b)
    b = b.replace('utils::is_zero(', 'is_zero(')
    b = b.replace('std::numeric_limits<double>::epsilon()', 'DBL_EPSILON')
    b = b.replace('std::array<double,3> f;', 'double f[3];')
    b = b.replace('std::array<double,3> ud{-f[0], -f[1], -f[2]};', 'double ud[3] = {-f[0], -f[1], -f[2]};')
    b = b.replace('std::array<double,3> d2u2{0.0};', 'double d2u2[3] = {0.0};')
    b = b.replace('std::array<double,3> mu_x{0.0};', 'double mu_x[3] = {0.0};')
    b = re.sub(r'std::transform\(mu_x.begin\(\), mu_x.end\(\), mu_x.begin\(\), \[mu_g\]\(double &v\)\{return mu_g\*v;\}\);',
               'for (int i = 0; i < 3; i++) mu_x[i] = mu_g * mu_x[i];', b)
    b = b.replace('std::memset(', 'memset(')
    assert 'com_mod' not in b and 'std::' not in b and 'dmn' not in b, name
    out.append('__device__ void d_%s(const FluidParams& prm, const int vmsFlag, const int eNoNw, const int eNoNq, const double w,\n'
               '    const double Kxi_[3][3], const double* Nw_, const double* Nq_, const double Nwx_[3][4],\n'
               '    const double Nqx_[3][4], const double Nwxx_[6][4], const double al_[4][4], const double yl_[4][4],\n'
               '    const double bfl_[3][4], double lR_[4][4], double lK_[16][4][4], double brinkman_inverse_permeability,\n'
               '    const double urisFactorTotal, const double* urisValveVelTermTotal)\n{' % name + b + '}\n')
open(sys.argv[2], 'w').write('\n'.join(out))

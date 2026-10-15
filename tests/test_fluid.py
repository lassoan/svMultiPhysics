from .conftest import run_with_reference, build_genbc, skip_if_no_petsc, skip_if_no_trilinos
import os

# Common folder for all tests in this file
base_folder = "fluid"

# Fields to test
fields = ["Velocity", "Pressure", "Traction", "WSS", "Vorticity", "Divergence"]


def test_pipe_RCR_3d(n_proc):
    test_folder = "pipe_RCR_3d"
    t_max = 2
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max)

def test_pipe_RCR_3d_fourier_coeff(n_proc):
    test_folder = "pipe_RCR_3d_fourier_coeff"
    t_max = 2
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max)

@skip_if_no_petsc
def test_pipe_RCR_3d_petsc(n_proc):
    test_folder = "pipe_RCR_3d_petsc"
    t_max = 2
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max)

@skip_if_no_trilinos
def test_pipe_RCR_3d_trilinos_ilut(n_proc):
    test_folder = "pipe_RCR_3d_trilinos_ilut"
    t_max = 2
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max)

@skip_if_no_trilinos
def test_pipe_RCR_3d_trilinos_bj(n_proc):
    test_folder = "pipe_RCR_3d_trilinos_bj"
    t_max = 2
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max)

def test_pipe_RCR_weak_dir_3d(n_proc):
    test_folder = "pipe_RCR_weak_dir_3d"
    t_max = 2
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max)

def test_pipe_RCR_genBC(n_proc):
    test_folder = "pipe_RCR_genBC"
    t_max = 2

    # Remove old genBC output and build genBC
    build_genbc(base_folder, test_folder, "genBC")

    run_with_reference(base_folder, test_folder, fields, n_proc, t_max)

def test_pipe_RCR_sv0D(n_proc):
    test_folder = "pipe_RCR_sv0D"
    t_max = 2
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max)


def test_driven_cavity_2d(n_proc):
    test_folder = "driven_cavity_2d"
    t_max = 2
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max)


def test_driven_cavity_2d_porous(n_proc):
    test_folder = "driven_cavity_2d_porous"
    t_max = 2
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max)


def test_dye_AD(n_proc):
    test_folder = "dye_AD"
    run_with_reference(base_folder, test_folder, ['Pressure', 'Velocity', 'Concentration'], n_proc)


def test_precomputed_dye_AD(n_proc):
    test_folder = "precomputed_dye_AD"
    run_with_reference(base_folder, test_folder, ['Velocity', 'Concentration'], n_proc)


def test_newtonian(n_proc):
    test_folder = "newtonian"
    run_with_reference(base_folder, test_folder, fields, n_proc)


def test_casson(n_proc):
    test_folder = "casson"
    run_with_reference(base_folder, test_folder, fields, n_proc)


def test_carreau_yasuda(n_proc):
    test_folder = "carreau_yasuda"
    run_with_reference(base_folder, test_folder, fields, n_proc)


def test_iliac_artery(n_proc):
    test_folder = "iliac_artery"
    run_with_reference(base_folder, test_folder, fields, n_proc)

@skip_if_no_trilinos
def test_iliac_artery_trilinos_gmres_ilut(n_proc):
    test_folder = "iliac_artery_trilinos_gmres_ilut"
    run_with_reference(base_folder, test_folder, fields, n_proc)

def test_quadratic_tet10(n_proc):
    test_folder = "quadratic_tet10"
    t_max = 3
    fields = ["Velocity", "Pressure", "Vorticity", "Divergence"]
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max) 

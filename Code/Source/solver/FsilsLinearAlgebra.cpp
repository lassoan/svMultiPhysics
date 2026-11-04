// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "FsilsLinearAlgebra.h"
#include "fsils_api.hpp"
#include "lhsa.h"
#include <iostream>
#ifdef SV_USE_MFEM
#include "svmfem_host.h"
#endif

/////////////////////////////////////////////////////////////////
//             F s i l s L i n e a r A l g e b r a             //
/////////////////////////////////////////////////////////////////
// The following methods implement the FSILS LinearAlgebra interface.

std::set<consts::LinearAlgebraType> FsilsLinearAlgebra::valid_assemblers = {
  consts::LinearAlgebraType::none,
  consts::LinearAlgebraType::fsils,
};

FsilsLinearAlgebra::FsilsLinearAlgebra()
{
  interface_type = consts::LinearAlgebraType::fsils; 
  assembly_type = consts::LinearAlgebraType::fsils; 
  preconditioner_type = consts::PreconditionerType::PREC_FSILS;
}

/// @brief Allocate data arrays.
void FsilsLinearAlgebra::alloc(ComMod& com_mod, eqType& lEq)
{
  #define n_debug_alloc
  #ifdef debug_alloc
  std::cout << "[FsilsLinearAlgebra::alloc] ---------- alloc ---------- " << std::endl;
  #endif
  int dof = com_mod.dof;

#ifdef SV_USE_MFEM
  // The matrix is assembled on the GPU (see assemble): the host matrix is
  // not used, so do not reallocate and clear it in every Newton iteration
  if (svmfem_host::val_on_device() && com_mod.Val.size() != 0) return;
#endif
  com_mod.Val.resize(dof*dof, com_mod.lhs.nnz);
}

/// @brief Assemble local element arrays.
void FsilsLinearAlgebra::assemble(ComMod& com_mod, const int num_elem_nodes, const Vector<int>& eqN,
        const Array3<double>& lK, const Array<double>& lR)
{
  #define n_debug_assemble
  #ifdef debug_assemble
  std::cout << "[FsilsLinearAlgebra::assemble] ---------- assemble ---------- " << std::endl;
  std::cout << "[FsilsLinearAlgebra::assemble] num_elem_nodes: " << num_elem_nodes << std::endl;
  std::cout << "[FsilsLinearAlgebra::assemble] eqN.size(): " << eqN.size() << std::endl;
  std::cout << "[FsilsLinearAlgebra::assemble] lK.size(): " << lK.size() << std::endl;
  std::cout << "[FsilsLinearAlgebra::assemble] lR.size(): " << lR.size() << std::endl;
  #endif

#ifdef SV_USE_MFEM
  if (svmfem_host::val_on_device()) {
    // The matrix is on the GPU: add the residual here and stage the matrix
    // entries (added to the device matrix in fsils_solve)
    auto& R = com_mod.R;
    const auto& rowPtr = com_mod.rowPtr;
    const auto& colPtr = com_mod.colPtr;
    for (int a = 0; a < num_elem_nodes; a++) {
      int rowN = eqN(a);
      if (rowN == -1) continue;
      for (int i = 0; i < R.nrows(); i++) R(i,rowN) = R(i,rowN) + lR(i,a);
      for (int b = 0; b < num_elem_nodes; b++) {
        int colN = eqN(b);
        if (colN == -1) continue;
        int left = rowPtr(rowN), right = rowPtr(rowN+1);
        int ptr = (right + left) / 2;
        while (colN != colPtr(ptr)) {
          if (colN > colPtr(ptr)) left = ptr; else right = ptr;
          ptr = (right + left) / 2;
        }
        svmfem_host::stage(ptr, &lK(0,a,b));
      }
    }
    return;
  }
#endif
  lhsa_ns::do_assem(com_mod, num_elem_nodes, eqN, lK, lR);
}

/// @brief Check the validity of the preconditioner and assembly types options. 
void FsilsLinearAlgebra::check_options(const consts::PreconditionerType prec_cond_type, 
  const consts::LinearAlgebraType assembly_type)
{
  using namespace consts;
  auto prec_cond_type_name = consts::preconditioner_type_to_name.at(prec_cond_type);
  auto assembly_type_name = LinearAlgebra::type_to_name.at(assembly_type);
  std::string error_msg;

  if (valid_assemblers.count(assembly_type) == 0) {
    error_msg = "fsils linear algebra can't use '" + assembly_type_name + "' for assembly.";
  }

  if (fsils_preconditioners.count(prec_cond_type) == 0) { 
    error_msg = "fsils linear algebra can't use '" + prec_cond_type_name + "' for a preconditioner.";
  }

  if (error_msg != "") { 
    throw std::runtime_error("[svMultiPhysics] ERROR: " + error_msg);
  }
}

/// @brief Initialize framework.
void FsilsLinearAlgebra::initialize(ComMod& com_mod, eqType& lEq)
{
  // Nothing is needed to initialize FSILS.
}

/// @brief Finalize framework.
void FsilsLinearAlgebra::finalize()
{
  // Nothing is needed to finalize FSILS.
}

/// @brief Set the linear algebra package for assmbly.
void FsilsLinearAlgebra::set_assembly(consts::LinearAlgebraType atype)
{
  if (atype == consts::LinearAlgebraType::none) {
    return;
  }

  if (valid_assemblers.count(atype) == 0) {
    auto str_type = LinearAlgebra::type_to_name.at(atype);
    throw std::runtime_error("[FsilsLinearAlgebra] ERROR: Can't set fsils linear algebra to use '" +
      str_type + "' for assembly.");
  }

  assembly_type = atype;
}

/// @brief Set the preconditioner.
void FsilsLinearAlgebra::set_preconditioner(consts::PreconditionerType prec_type)
{
  if (consts::fsils_preconditioners.count(prec_type) == 0) {
    auto prec_cond_type_name = consts::preconditioner_type_to_name.at(prec_type);
    throw std::runtime_error("[FsilsLinearAlgebra] ERROR: fsils linear algebra can't use '" + 
        prec_cond_type_name + "' for a preconditioner.");
    return;
  }

  preconditioner_type = prec_type;
}

/// @brief Solve a system of linear equations.
void FsilsLinearAlgebra::solve(ComMod& com_mod, eqType& lEq, const Vector<int>& incL, const Vector<double>& res)
{
  auto& lhs = com_mod.lhs;
  int dof = com_mod.dof;
  auto& R = com_mod.R;      
  auto& Val = com_mod.Val;
  auto preconditioner = lEq.linear_algebra_preconditioner;

  fsi_linear_solver::fsils_solve(lhs, lEq.FSILS, dof, R, Val, preconditioner, incL, res);
}


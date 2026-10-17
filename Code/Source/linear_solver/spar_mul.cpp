// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

//--------------------------------------------------------------------
// Product of a sparse matrix and a vector. The matrix might be
// vector in neither, one or both dimensions.
//--------------------------------------------------------------------
//
// Reproduces code in SPARMUL.f.

#include "spar_mul.h"

#include "fsils_api.hpp"

namespace spar_mul {

/// @brief Reproduces 'SUBROUTINE FSILS_SPARMULSS(lhs, rowPtr, colPtr, K, U, KU)'
//
void fsils_spar_mul_ss(FSILS_lhsType& lhs, const Array<int>& rowPtr, const Vector<int>& colPtr, 
    const Vector<double>& K, const Vector<double>& U, Vector<double>& KU)
{
  int nNo = lhs.nNo;
  KU = 0.0;

  for (int i = 0; i < nNo; i++) {
    double s = 0.0;
    for (int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
      s = s + K(j) * U(colPtr(j));
    } 
    KU(i) = s;
  }

  fsils_commus(lhs, KU);
}

/// @brief Reproduces 'SUBROUTINE FSILS_SPARMULSV(lhs, rowPtr, colPtr, dof, K, U, KU)'. 
//
void fsils_spar_mul_sv(FSILS_lhsType& lhs, const Array<int>& rowPtr, const Vector<int>& colPtr, 
    const int dof, const Array<double>& K, const Vector<double>& U, Array<double>& KU)
{
  int nNo = lhs.nNo;
  KU = 0.0;

  switch (dof) {

    case 1:
      for (int i = 0; i < nNo; i++) {
        double s0 = 0.0;
        for (int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
          s0 = s0 + K(0,j)*U(colPtr(j));
        }
        KU(0,i) = s0;
      }
    break; 

    case 2: {
      for (int i = 0; i < nNo; i++) {
        double s0 = 0.0, s1 = 0.0;
        for (int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
          int col = colPtr(j);
          s0 = s0 + K(0,j)*U(col);
          s1 = s1 + K(1,j)*U(col);
        }
        KU(0,i) = s0;
        KU(1,i) = s1;
      }

    } break; 

    case 3: {
      for (int i = 0; i < nNo; i++) {
        double s0 = 0.0, s1 = 0.0, s2 = 0.0;
        for (int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
          int col = colPtr(j);
          s0 += K(0,j) * U(col);
          s1 += K(1,j) * U(col);
          s2 += K(2,j) * U(col);
        }
        KU(0,i) = s0;
        KU(1,i) = s1;
        KU(2,i) = s2;
      }
    } break; 

    case 4:
      for (int i = 0; i < nNo; i++) {
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        for (int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
          int col = colPtr(j);
          s0 = s0 + K(0,j)*U(col);
          s1 = s1 + K(1,j)*U(col);
          s2 = s2 + K(2,j)*U(col);
          s3 = s3 + K(3,j)*U(col);
        }
        KU(0,i) = s0;
        KU(1,i) = s1;
        KU(2,i) = s2;
        KU(3,i) = s3;
      }
    break; 

    default: 
      for (int i = 0; i < nNo; i++) {
        for (int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
          int col = colPtr(j);
          for (int m = 0; m < KU.nrows(); m++) {
            KU(m,i) = KU(m,i) + K(m,j) * U(col);
          }
        }
      }
  } 

  fsils_commuv(lhs, dof, KU);
}

/// @brief Reproduces 'SUBROUTINE FSILS_SPARMULVS(lhs, rowPtr, colPtr, dof, K, U, KU)'.
//
void fsils_spar_mul_vs(FSILS_lhsType& lhs, const Array<int>& rowPtr, const Vector<int>& colPtr, 
    const int dof, const Array<double>& K, const Array<double>& U, Vector<double>& KU)
{
  int nNo = lhs.nNo;
  KU = 0.0;

  switch (dof) {

    case 1:
      for (int i = 0; i < nNo; i++) {
        double s = 0.0;
        for (int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
          s = s + K(0,j) * U(0,colPtr(j));
        }
        KU(i) = s;
      }
    break; 

    case 2:
      for (int i = 0; i < nNo; i++) {
        double s = 0.0;
        for (int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
          int col = colPtr(j);
          s = s + K(0,j)*U(0,col) + K(1,j)*U(1,col);
        }
        KU(i) = s;
      }
    break; 

    case 3:
      for (int i = 0; i < nNo; i++) {
        double s = 0.0;
        for (int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
          int col = colPtr(j);
          s = s + K(0,j)*U(0,col) + K(1,j)*U(1,col) + K(2,j)*U(2,col);
        }
        KU(i) = s;
      }
    break; 

    case 4:
      for (int i = 0; i < nNo; i++) {
        double s = 0.0;
        for (int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
          int col = colPtr(j);
          s = s + K(0,j)*U(0,col) + K(1,j)*U(1,col) + K(2,j)*U(2,col) + K(3,j)*U(3,col);
        }
        KU(i) = s;
      }
    break; 

    default: 
      for (int i = 0; i < nNo; i++) {
        double s = 0.0;
        for (int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
          int col = colPtr(j);
          double sum = 0.0;
          for (int m = 0; m < K.nrows(); m++) {
            sum += K(m,j) * U(m,col);
          }
          s = s + sum; 
          //KU(i) = KU(i) + SUM(K(:,j)*U(:,colPtr(j)))
        }
        KU(i) = s;
     }
  } 

  fsils_commus(lhs, KU);
}

/// @brief Reproduces 'SUBROUTINE FSILS_SPARMULVV(lhs, rowPtr, colPtr, dof, K, U, KU)'. 
//
void fsils_spar_mul_vv(FSILS_lhsType& lhs, const Array<int>& rowPtr, const Vector<int>& colPtr, 
    const int dof, const Array<double>& K, const Array<double>& U, Array<double>& KU)
{
  int nNo = lhs.nNo;
  KU = 0.0;

  switch (dof) {

    case 1:
      for (int i = 0; i < nNo; i++) {
        double s0 = 0.0;
        for (int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
          s0 = s0 + K(0,j)*U(0,colPtr(j));
        }
        KU(0,i) = s0;
      }
    break;

    case 2:
      for (int i = 0; i < nNo; i++) {
        double s0 = 0.0, s1 = 0.0;
        for (int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
          int col = colPtr(j);
          s0 = s0 + K(0,j)*U(0,col) + K(1,j)*U(1,col);
          s1 = s1 + K(2,j)*U(0,col) + K(3,j)*U(1,col);
        }
        KU(0,i) = s0;
        KU(1,i) = s1;
      }
    break;

    case 3:
      for (int i = 0; i < nNo; i++) {
        double s0 = 0.0, s1 = 0.0, s2 = 0.0;
        for (int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
          int col = colPtr(j);
          s0 = s0 + K(0,j)*U(0,col) + K(1,j)*U(1,col) + K(2,j)*U(2,col);
          s1 = s1 + K(3,j)*U(0,col) + K(4,j)*U(1,col) + K(5,j)*U(2,col);
          s2 = s2 + K(6,j)*U(0,col) + K(7,j)*U(1,col) + K(8,j)*U(2,col);
        }
        KU(0,i) = s0;
        KU(1,i) = s1;
        KU(2,i) = s2;
      }
    break;

    case 4:
      for (int i = 0; i < nNo; i++) {
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        for (int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
          int col = colPtr(j);
          s0 = s0 + K(0 ,j)*U(0,col) + K(1 ,j)*U(1,col) + K(2 ,j)*U(2,col) + K(3 ,j)*U(3,col);
          s1 = s1 + K(4 ,j)*U(0,col) + K(5 ,j)*U(1,col) + K(6 ,j)*U(2,col) + K(7 ,j)*U(3,col);
          s2 = s2 + K(8 ,j)*U(0,col) + K(9,j)*U(1,col) + K(10,j)*U(2,col) + K(11,j)*U(3,col);
          s3 = s3 + K(12,j)*U(0,col) + K(13,j)*U(1,col) + K(14,j)*U(2,col) + K(15,j)*U(3,col);
        }
        KU(0,i) = s0;
        KU(1,i) = s1;
        KU(2,i) = s2;
        KU(3,i) = s3;
      }
    break;

    default: 
      for (int i = 0; i < nNo; i++) {
        for (int j = rowPtr(0,i); j <= rowPtr(1,i); j++) {
          int col = colPtr(j);
          for (int l = 0; l < dof; l++) {
            int e = l*dof;
            int s = e - dof + 1;;
            double sum = 0.0;
            for (int k = 0; k < dof; k++) {
              sum += K(k+s,j) * U(k,col);
            }
            KU(l,i) = KU(l,i) + sum;
          }
        }
     }
  } 

  fsils_commuv(lhs, dof, KU);
}

};



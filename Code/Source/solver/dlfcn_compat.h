// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DLFCN_COMPAT_H
#define DLFCN_COMPAT_H

/// @brief Portable access to dlopen/dlsym/dlclose/dlerror.
///
/// On Windows the functions are implemented using LoadLibrary/GetProcAddress.
//
#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <cstdio>
// Windows headers may define 'interface' as a macro, which clashes with variable names
#undef interface

#ifndef RTLD_LAZY
#define RTLD_LAZY 0
#endif

inline const char* dlerror_message(bool set, DWORD code = 0)
{
  static thread_local char buf[512];
  static thread_local bool has_error = false;
  if (set) {
    has_error = true;
    DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, 0, buf, sizeof(buf), nullptr);
    if (n == 0) {
      snprintf(buf, sizeof(buf), "Windows error %lu", static_cast<unsigned long>(code));
    }
    return buf;
  }
  if (!has_error) {
    return nullptr;
  }
  has_error = false;
  return buf;
}

inline void* dlopen(const char* file, int /*mode*/)
{
  HMODULE handle = LoadLibraryA(file);
  if (!handle) {
    dlerror_message(true, GetLastError());
  }
  return reinterpret_cast<void*>(handle);
}

inline void* dlsym(void* handle, const char* name)
{
  FARPROC sym = GetProcAddress(reinterpret_cast<HMODULE>(handle), name);
  if (!sym) {
    dlerror_message(true, GetLastError());
  }
  return reinterpret_cast<void*>(sym);
}

inline int dlclose(void* handle)
{
  return FreeLibrary(reinterpret_cast<HMODULE>(handle)) ? 0 : -1;
}

inline const char* dlerror()
{
  return dlerror_message(false);
}

#else

#include <dlfcn.h>

#endif

#endif

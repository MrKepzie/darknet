/* ***** BEGIN LICENSE BLOCK *****
 * Copyright (C) 2017 INRIA
 * Author: Alexandre Gauthier-Foichat
 * ***** END LICENSE BLOCK ***** */


#ifndef FSTREAMSSUPPORT_H
#define FSTREAMSSUPPORT_H


#include <string>
#include <fstream>
#if defined(_WIN32) && defined(__GLIBCXX__)
#define FSTREAM_USE_STDIO_FILEBUF 1
#include "fstream_mingw.h"
#endif


namespace FStreamsSupport {
#if FSTREAM_USE_STDIO_FILEBUF
// MingW uses GCC to build, but does not support having a wchar_t* passed as argument
// of ifstream::open or ofstream::open. To properly support UTF-8 encoding on MingW we must
// use the __gnu_cxx::stdio_filebuf GNU extension that can be used with _wfsopen and returned
// into a istream which share the same API as ifsteam. The same reasoning holds for ofstream.
typedef basic_ifstream<char> ifstream;
typedef basic_ofstream<char> ofstream;
#else
typedef std::ifstream ifstream;
typedef std::ofstream ofstream;
#endif


void open(ifstream* stream, const std::string& filename, std::ios_base::openmode mode = std::ios_base::in);

void open(ofstream* stream, const std::string& filename, std::ios_base::openmode mode = std::ios_base::out);
} //FStreamsSupport


#endif // FSTREAMSSUPPORT_H

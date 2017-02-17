/* ***** BEGIN LICENSE BLOCK *****
 * Copyright (C) 2017 INRIA
 * Author: Alexandre Gauthier-Foichat
 * ***** END LICENSE BLOCK ***** */


#include "FStreamsSupport.h"
#include "StrUtils.h"



void
FStreamsSupport::open (FStreamsSupport::ifstream *stream,
                       const std::string& path,
                       std::ios_base::openmode mode)
{
#ifdef _WIN32
    // Windows std::ifstream accepts non-standard wchar_t*
    // On MingW, we use our own FStreamsSupport::ifstream
    std::wstring wpath = StrUtils::utf8_to_utf16(path);
    stream->open (wpath.c_str(), mode);
    stream->seekg (0, std::ios_base::beg); // force seek, otherwise broken
#else
    stream->open (path.c_str(), mode);
#endif
}

void
FStreamsSupport::open (FStreamsSupport::ofstream *stream,
                       const std::string& path,
                       std::ios_base::openmode mode)
{
#ifdef _WIN32
    // Windows std::ofstream accepts non-standard wchar_t*
    // On MingW, we use our own FStreamsSupport::ofstream
    std::wstring wpath = StrUtils::utf8_to_utf16(path);
    stream->open (wpath.c_str(), mode);
#else
    stream->open (path.c_str(), mode);
#endif
}


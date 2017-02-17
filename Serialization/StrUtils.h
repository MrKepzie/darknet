/* ***** BEGIN LICENSE BLOCK *****
 * Copyright (C) 2017 INRIA
 * Author: Alexandre Gauthier-Foichat
 * ***** END LICENSE BLOCK ***** */

#ifndef STRUTILS_H
#define STRUTILS_H

#include <string>

namespace StrUtils {

// Should be used in asserts to ensure strings are utf8
bool is_utf8(const char * string);

/*Converts a std::string to wide string*/
std::wstring utf8_to_utf16(const std::string & s);

std::string utf16_to_utf8 (const std::wstring& str);

#ifdef _WIN32
//Returns the last Win32 error, in string format. Returns an empty string if there is no error.
std::string GetLastErrorAsString();
#endif // __NATRON_WIN32__


} // namespace StrUtils


#endif // ifndef STRUTILS_H

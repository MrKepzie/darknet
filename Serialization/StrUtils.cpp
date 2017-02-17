/* ***** BEGIN LICENSE BLOCK *****
 * Copyright (C) 2017 INRIA
 * Author: Alexandre Gauthier-Foichat
 * ***** END LICENSE BLOCK ***** */

#include "StrUtils.h"

#include <utility>
#if defined(_WIN32)
#include <string>
#include <windows.h>
#include <fcntl.h>
#include <sys/stat.h>
#else
#include <cstdlib>
#include <climits>
#endif


namespace StrUtils {

    bool is_utf8(const char * string)
    {
        if(!string)
            return false;

        const unsigned char * bytes = (const unsigned char *)string;
        while(*bytes)
        {
            if( (// ASCII
                 // use bytes[0] <= 0x7F to allow ASCII control characters
                 bytes[0] == 0x09 ||
                 bytes[0] == 0x0A ||
                 bytes[0] == 0x0D ||
                 (0x20 <= bytes[0] && bytes[0] <= 0x7E)
                 )
               ) {
                bytes += 1;
                continue;
            }

            if( (// non-overlong 2-byte
                 (0xC2 <= bytes[0] && bytes[0] <= 0xDF) &&
                 (0x80 <= bytes[1] && bytes[1] <= 0xBF)
                 )
               ) {
                bytes += 2;
                continue;
            }

            if( (// excluding overlongs
                 bytes[0] == 0xE0 &&
                 (0xA0 <= bytes[1] && bytes[1] <= 0xBF) &&
                 (0x80 <= bytes[2] && bytes[2] <= 0xBF)
                 ) ||
               (// straight 3-byte
                ((0xE1 <= bytes[0] && bytes[0] <= 0xEC) ||
                 bytes[0] == 0xEE ||
                 bytes[0] == 0xEF) &&
                (0x80 <= bytes[1] && bytes[1] <= 0xBF) &&
                (0x80 <= bytes[2] && bytes[2] <= 0xBF)
                ) ||
               (// excluding surrogates
                bytes[0] == 0xED &&
                (0x80 <= bytes[1] && bytes[1] <= 0x9F) &&
                (0x80 <= bytes[2] && bytes[2] <= 0xBF)
                )
               ) {
                bytes += 3;
                continue;
            }

            if( (// planes 1-3
                 bytes[0] == 0xF0 &&
                 (0x90 <= bytes[1] && bytes[1] <= 0xBF) &&
                 (0x80 <= bytes[2] && bytes[2] <= 0xBF) &&
                 (0x80 <= bytes[3] && bytes[3] <= 0xBF)
                 ) ||
               (// planes 4-15
                (0xF1 <= bytes[0] && bytes[0] <= 0xF3) &&
                (0x80 <= bytes[1] && bytes[1] <= 0xBF) &&
                (0x80 <= bytes[2] && bytes[2] <= 0xBF) &&
                (0x80 <= bytes[3] && bytes[3] <= 0xBF)
                ) ||
               (// plane 16
                bytes[0] == 0xF4 &&
                (0x80 <= bytes[1] && bytes[1] <= 0x8F) &&
                (0x80 <= bytes[2] && bytes[2] <= 0xBF) &&
                (0x80 <= bytes[3] && bytes[3] <= 0xBF)
                )
               ) {
                bytes += 4;
                continue;
            }
            
            return false;
        }
        
        return true;
    } // is_utf8
    
    /*Converts a std::string to wide string*/
    std::wstring
            utf8_to_utf16(const std::string & str)
    {
#ifdef _WIN32
        std::wstring native;


        native.resize(MultiByteToWideChar (CP_UTF8, 0, str.data(), str.length(), NULL, 0));
        MultiByteToWideChar ( CP_UTF8, 0, str.data(), str.length(), &native[0], (int)native.size() );

        return native;

#else
        std::wstring dest;
        size_t max = str.size() * 4;
        mbtowc (NULL, NULL, max);  /* reset mbtowc */

        const char* cstr = str.c_str();

        while (max > 0) {
            wchar_t w;
            size_t length = mbtowc(&w, cstr, max);
            if (length < 1) {
                break;
            }
            dest.push_back(w);
            cstr += length;
            max -= length;
        }

        return dest;
#endif
    } // utf8_to_utf16

    std::string
            utf16_to_utf8 (const std::wstring& str)
    {
#ifdef _WIN32
        std::string utf8;

        utf8.resize(WideCharToMultiByte (CP_UTF8, 0, str.data(), str.length(), NULL, 0, NULL, NULL));
        WideCharToMultiByte (CP_UTF8, 0, str.data(), str.length(), &utf8[0], (int)utf8.size(), NULL, NULL);

        return utf8;
#else
        std::string utf8;
        for (std::size_t i = 0; i < str.size(); ++i) {
            char c[MB_LEN_MAX];
            int nbBytes = wctomb(c, str[i]);
            if (nbBytes > 0) {
                for (int j = 0; j < nbBytes; ++j) {
                    utf8.push_back(c[j]);
                }
            } else {
                break;
            }
        }
        return utf8;
#endif
    } // utf16_to_utf8


#ifdef _WIN32


    //Returns the last Win32 error, in string format. Returns an empty string if there is no error.
    std::string
            GetLastErrorAsString()
    {
        //Get the error message, if any.
        DWORD errorMessageID = ::GetLastError();

        if (errorMessageID == 0) {
            return std::string(); //No error message has been recorded
        }
        LPSTR messageBuffer = 0;
        size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                     NULL, errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);
        std::string message(messageBuffer, size);

        //Free the buffer.
        LocalFree(messageBuffer);

        return message;
    } // GetLastErrorAsString

#endif // _WIN32
}


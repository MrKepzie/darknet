#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include <sys/time.h>

#include "SequenceParsing.h"
#include <list>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <iostream>
#include <vector>
#include "DetectionsSerialization.h"
#include "SerializationIO.h"
#include "FStreamsSupport.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#endif

// The size of the model will be HUE_HISTOGRAM_NUM_BINS * SAT_HISTOGRAM_NUM_BINS
#define HUE_HISTOGRAM_NUM_BINS 32
#define SAT_HISTOGRAM_NUM_BINS 32

#define MAX_NUM_MODELS_PER_FILE 100000

class RectI
{
public:

    int x1; // left
    int y1; // bottom
    int x2; // right
    int y2; // top

    RectI()
    : x1(0), y1(0), x2(0), y2(0)
    {
    }

    int width() const
    {
        return x2 - x1;
    }

    int height() const
    {
        return y2 - y1;
    }

    /// returns true if the rect passed as parameter  intersects this one
    bool intersects(const RectI & r) const
    {
        if ( isNull() || r.isNull() ) {
            return false;
        }
        if ( (r.x2 <= x1) || (x2 <= r.x1) || (r.y2 <= y1) || (y2 <= r.y1) ) {
            return false;
        }

        return true;
    }

    bool intersect(const RectI & r,
                   RectI* intersection) const
    {
        if ( !intersects(r) ) {
            return false;
        }

        intersection->x1 = std::max(x1, r.x1);
        intersection->y1 = std::max(y1, r.y1);
        // the region must be *at least* empty, thus the maximin.
        intersection->x2 = std::max( intersection->x1, std::min(x2, r.x2) );
        // the region must be *at least* empty, thus the maximin.
        intersection->y2 = std::max( intersection->y1, std::min(y2, r.y2) );

        assert( !intersection->isNull() );

        return true;
    }


    bool isNull() const
    {
        return (x2 <= x1) || (y2 <= y1);
    }

    bool contains(const RectI & other) const
    {
        return other.x1 >= x1 &&
        other.y1 >= y1 &&
        other.x2 <= x2 &&
        other.y2 <= y2;
    }
};

class RectD
{
public:

    double x1; // left
    double y1; // bottom
    double x2; // right
    double y2; // top

    RectD()
    : x1(0), y1(0), x2(0), y2(0)
    {
    }

    RectD(double x1, double y1, double x2, double y2)
    : x1(x1), y1(y1), x2(x2), y2(y2)
    {
    }



    double width() const
    {
        return x2 - x1;
    }

    double height() const
    {
        return y2 - y1;
    }

    bool intersect(const RectD & r,
                   RectD* intersection) const
    {
        if ( isNull() || r.isNull() ) {
            return false;
        }

        if ( (x1 > r.x2) || (r.x1 > x2) || (y1 > r.y2) || (r.y1 > y2) ) {
            return false;
        }

        intersection->x1 = std::max(x1, r.x1);
        intersection->x2 = std::min(x2, r.x2);
        intersection->y1 = std::max(y1, r.y1);
        intersection->y2 = std::min(y2, r.y2);

        return true;
    }

    bool isNull() const
    {
        return (x2 <= x1) || (y2 <= y1);
    }
};

template <typename PIX>
inline const PIX* getPixels(int x, int y, const RectI& bounds, int nComps, const PIX* pixelData)
{
    if ( ( x < bounds.x1 ) || ( x >= bounds.x2 ) || ( y < bounds.y1 ) || ( y >= bounds.y2 ) ) {
        return NULL;
    } else {
        const unsigned char* ret = (const unsigned char*)pixelData;
        if (!ret) {
            return 0;
        }

        std::size_t compDataSize = sizeof(PIX) * nComps;
        ret = ret + (std::size_t)( y - bounds.y1 ) * compDataSize * bounds.width()
        + (std::size_t)( x - bounds.x1 ) * compDataSize;

        return (const PIX*)ret;
    }
}

template <typename PIX>
inline PIX* getPixels(int x, int y, const RectI& bounds, int nComps, PIX* pixelData)
{
    if ( ( x < bounds.x1 ) || ( x >= bounds.x2 ) || ( y < bounds.y1 ) || ( y >= bounds.y2 ) ) {
        return NULL;
    } else {
        unsigned char* ret = (unsigned char*)pixelData;
        if (!ret) {
            return 0;
        }

        std::size_t compDataSize = sizeof(PIX) * nComps;
        ret = ret + (std::size_t)( y - bounds.y1 ) * compDataSize * bounds.width()
        + (std::size_t)( x - bounds.x1 ) * compDataSize;

        return (PIX*)ret;
    }
}

template <typename SRCPIX, typename DSTPIX>
inline DSTPIX convertPixelDepth(SRCPIX pix);


/// numvals should be 256 for byte, 65536 for 16-bits, etc.

/// maps 0-(numvals-1) to 0.-1.
template<int numvals>
inline float
intToFloat(int value)
{
    return value / (float)(numvals - 1);
}

/// maps 0.-1. to 0-(numvals-1)
template<int numvals>
inline int
floatToInt(float value)
{
    if (value <= 0) {
        return 0;
    } else if (value >= 1.) {
        return numvals - 1;
    }

    return value * (numvals - 1) + 0.5;
}

///explicit template instantiations

template <>
float
convertPixelDepth(unsigned char pix)
{
    return intToFloat<256>(pix);
}

template <>
unsigned short
convertPixelDepth(unsigned char pix)
{
    // 0x01 -> 0x0101, 0x02 -> 0x0202, ..., 0xff -> 0xffff
    return (unsigned short)( (pix << 8) + pix );
}

template <>
unsigned char
convertPixelDepth(unsigned char pix)
{
    return pix;
}

template <>
unsigned char
convertPixelDepth(unsigned short pix)
{
    // the following is from ImageMagick's quantum.h
    return (unsigned char)( ( (pix + 128UL) - ( (pix + 128UL) >> 8 ) ) >> 8 );
}

template <>
float
convertPixelDepth(unsigned short pix)
{
    return intToFloat<65536>(pix);
}

template <>
unsigned short
convertPixelDepth(unsigned short pix)
{
    return pix;
}

template <>
unsigned char
convertPixelDepth(float pix)
{
    return (unsigned char)floatToInt<256>(pix);
}

template <>
unsigned short
convertPixelDepth(float pix)
{
    return (unsigned short)floatToInt<65536>(pix);
}

template <>
float
convertPixelDepth(float pix)
{
    return pix;
}

// r,g,b values are from 0 to 1
// h = [0,1.], s = [0,1], v = [0,1]
//		if s == 0, then h = 0 (undefined)
inline void
rgb_to_hsv(float r,
           float g,
           float b,
           float *h,
           float *s,
           float *v )
{
    float min = std::min(std::min(r, g), b);
    float max = std::max(std::max(r, g), b);

    *v = max;                               // v

    float delta = max - min;

    if (max != 0.) {
        *s = delta / max;                       // s
    } else {
        // r = g = b = 0		// s = 0, v is undefined
        *s = 0.f;
        *h = 0.f;

        return;
    }

    if (delta == 0.) {
        *h = 0.f;                 // gray
    } else if (r == max) {
        *h = (g - b) / delta;                       // between yellow & magenta
    } else if (g == max) {
        *h = 2 + (b - r) / delta;                   // between cyan & yellow
    } else {
        *h = 4 + (r - g) / delta;                   // between magenta & cyan
    }
    *h *= 1. / 6.;
    if (*h < 0) {
        *h += 1.;
    }
}

enum BitDepthEnum
{
    eBitDepthByte,
    eBitDepthShort,
    eBitDepthFloat,
};

template <typename SRCPIX, int nComps, typename DSTPIX>
inline void convertRGBToHSVInternal(const SRCPIX* srcPixelsData, const RectI& srcBounds, const RectI& roi, DSTPIX* dstPixelsData, const RectI& dstBounds)
{

    const SRCPIX* srcPixels = getPixels<SRCPIX>(roi.x1, roi.y1, srcBounds, nComps, srcPixelsData);
    DSTPIX* dstPixels = getPixels<DSTPIX>(roi.x1, roi.y1, dstBounds, nComps, dstPixelsData);

    for (int y = roi.y1; y < roi.y2; ++y) {
        for (int x = roi.x1; x < roi.x2; ++x, srcPixels += nComps, dstPixels += nComps) {

            float tmpPix[3];
            memset(tmpPix, 0, sizeof(float) * 3);

            for (int c = 0; c < std::min(3,nComps); ++c) {
                tmpPix[c] = convertPixelDepth<SRCPIX, float>(srcPixels[c]);
            }
            float hsv[3];
            rgb_to_hsv(tmpPix[0], tmpPix[1], tmpPix[2], &hsv[0], &hsv[1], &hsv[2]);

            for (int c = 0; c < std::min(3,nComps); ++c) {
                dstPixels[c] = convertPixelDepth<float, DSTPIX>(hsv[c]);
            }
            for (int c = 3; c < nComps; ++c) {
                dstPixels[c] = convertPixelDepth<SRCPIX, DSTPIX>(srcPixels[c]);
            }

        }
        // Remove what was done on last iteration and go to the next line
        srcPixels += (srcBounds.width() * nComps - roi.width() * nComps);
        dstPixels += (dstBounds.width() * nComps - roi.width() * nComps);
    }

}

template <typename SRCPIX, int nComps>
inline void convertRGBToHSVForNComps(const SRCPIX* srcPixelsData, const RectI& srcBounds, const RectI& roi, BitDepthEnum dstDepth, void* dstPixelsData, const RectI& dstBounds)
{
    switch (dstDepth) {
        case eBitDepthByte:
            convertRGBToHSVInternal<SRCPIX, nComps, unsigned char>(srcPixelsData, srcBounds, roi, (unsigned char*)dstPixelsData, dstBounds);
            break;
        case eBitDepthShort:
            convertRGBToHSVInternal<SRCPIX, nComps, unsigned short>(srcPixelsData, srcBounds, roi, (unsigned short*)dstPixelsData, dstBounds);
            break;
        case eBitDepthFloat:
            convertRGBToHSVInternal<SRCPIX, nComps, float>(srcPixelsData, srcBounds, roi, (float*)dstPixelsData, dstBounds);
            break;
    }
}

template <typename SRCPIX>
inline void convertRGBToHSVForSrcBitDepth(int nComps, const SRCPIX* srcPixelsData, const RectI& srcBounds, const RectI& roi, BitDepthEnum dstDepth, void* dstPixelsData, const RectI& dstBounds)
{
    switch (nComps) {
        case 1:
            convertRGBToHSVForNComps<SRCPIX, 1>(srcPixelsData, srcBounds, roi, dstDepth, dstPixelsData, dstBounds);
            break;
        case 2:
            convertRGBToHSVForNComps<SRCPIX, 2>(srcPixelsData, srcBounds, roi, dstDepth, dstPixelsData, dstBounds);
            break;
        case 3:
            convertRGBToHSVForNComps<SRCPIX, 3>(srcPixelsData, srcBounds, roi, dstDepth, dstPixelsData, dstBounds);
            break;
        case 4:
            convertRGBToHSVForNComps<SRCPIX, 4>(srcPixelsData, srcBounds, roi, dstDepth, dstPixelsData, dstBounds);
            break;
        default:
            assert(false);
            break;
    }
}

inline void convertRGBToHSV(int nComps, BitDepthEnum srcDepth, const void* srcPixelsData, const RectI& srcBounds, const RectI& roi, BitDepthEnum dstDepth, void* dstPixelsData, const RectI& dstBounds)
{
    switch (srcDepth) {
        case eBitDepthByte:
            convertRGBToHSVForSrcBitDepth<unsigned char>(nComps, (const unsigned char*)srcPixelsData, srcBounds, roi, dstDepth, dstPixelsData, dstBounds);
            break;
        case eBitDepthShort:
            convertRGBToHSVForSrcBitDepth<unsigned short>(nComps, (const unsigned short*)srcPixelsData, srcBounds, roi, dstDepth, dstPixelsData, dstBounds);
            break;
        case eBitDepthFloat:
            convertRGBToHSVForSrcBitDepth<float>(nComps, (const float*)srcPixelsData, srcBounds, roi, dstDepth, dstPixelsData, dstBounds);
            break;
    }
}

/**
 * @brief Compute a N-Dimensional histogram where N <= srcNComps over the given region of interest.
 * The number of bins across each dimension is given by histogramSizes which must be of size srcNComps
 * If a dimension is of size 0, it indicates we are not interested in its histogram
 **/
template <typename PIX, int maxValue, int srcNComps>
inline void computeHistogramInternal(const PIX* srcPixelsData, const RectI& srcBounds, const RectI& roi, const std::vector<int>& histogramSizes, std::vector<double>* histogram)
{
    assert(srcBounds.contains(roi));
    assert(histogramSizes.size() == srcNComps);

    int histStrides[srcNComps];
    for (int i = 0; i < srcNComps; ++i) {
        if (histogramSizes[i] > 0) {
            histStrides[i] = 1;
        } else {
            histStrides[i] = 0;
        }

    }
    std::size_t numElements = 1;
    for (std::size_t i = 0; i < histogramSizes.size(); ++i) {
        if (histogramSizes[i] > 0) {
            numElements *= histogramSizes[i];
            if (i > 0) {
                for (std::size_t j = 0; j < i ; ++j) {
                    histStrides[j] *= histogramSizes[j];
                }
            }
        }
    }
    // Initialize to 0
    histogram->resize(numElements, 0);




    const PIX* srcPixels = getPixels(roi.x1, roi.y1, srcBounds, srcNComps, srcPixelsData);

    for (int y = roi.y1; y < roi.y2; ++y) {
        for (int x = roi.x1; x < roi.x2; ++x, srcPixels += srcNComps) {

            int indices[srcNComps];
            for (int c = 0; c < srcNComps; ++c) {
                if (histogramSizes[c]) {
                    // Convert to float and clamp to [0,1]
                    float val = std::max(0.f,std::min(1.f,convertPixelDepth<PIX, float>(srcPixels[c])));

                    // Map the normalized [0,1] value to a value in the range of the histogram size for that bin and round to nearest integer
                    indices[c] = std::floor(val * (histogramSizes[c] - 1) + 0.5);
                }
            }
            int index = 0;
            for (int c = 0; c < srcNComps; ++c) {
                if (histStrides[c]) {
                    index += histStrides[c] * indices[c];
                }
            }
            assert(index < (int)histogram->size());
            (*histogram)[index] += 1;
        }
        // Remove what was done on last iteration and go to the next line
        srcPixels += (srcBounds.width() * srcNComps - roi.width() * srcNComps);
    }
} // computeHistogramInternal

template <typename PIX, int maxValue>
inline void computeHistogramForNComps(int nComps, const PIX* srcPixelsData, const RectI& srcBounds, const RectI& roi, const std::vector<int>& histogramSizes, std::vector<double>* histogram)
{
    switch (nComps) {
        case 1:
            computeHistogramInternal<PIX, maxValue, 1>(srcPixelsData, srcBounds, roi, histogramSizes, histogram);
            break;
        case 2:
            computeHistogramInternal<PIX, maxValue, 2>(srcPixelsData, srcBounds, roi, histogramSizes, histogram);
            break;
        case 3:
            computeHistogramInternal<PIX, maxValue, 3>(srcPixelsData, srcBounds, roi, histogramSizes, histogram);
            break;
        case 4:
            computeHistogramInternal<PIX, maxValue, 4>(srcPixelsData, srcBounds, roi, histogramSizes, histogram);
            break;
    }

}

inline void computeHistogram(int nComps, BitDepthEnum depth, const void* srcPixelsData, const RectI& srcBounds, const RectI& roi, const std::vector<int>& histogramSizes, std::vector<double>* histogram)
{
    switch (depth) {
        case eBitDepthByte:
            computeHistogramForNComps<unsigned char, 255>(nComps, (const unsigned char*)srcPixelsData, srcBounds, roi, histogramSizes, histogram);
            break;
        case eBitDepthShort:
            computeHistogramForNComps<unsigned short, 65535>(nComps, (const unsigned short*)srcPixelsData, srcBounds, roi, histogramSizes, histogram);
            break;
        case eBitDepthFloat:
            computeHistogramForNComps<float, 1>(nComps, (const float*)srcPixelsData, srcBounds, roi, histogramSizes, histogram);
            break;
    }

}


enum HistogramComparisonEnum
{
    // the higher the metric, the more accurate the match
    eHistogramComparisonCorrelation,

    // the less the result, the better the match
    eHistogramComparisonChiSquare,

    // the higher the metric, the more accurate the match
    eHistogramComparisonIntersection,

    //  the less the result, the better the match
    eHistogramComparisonBhattacharyya,
    eHistogramComparisonKLDiv,
};

inline double compareHist(const std::vector<double>& h1, const std::vector<double>& h2, HistogramComparisonEnum method)
{
    assert(h1.size() == h2.size() && !h1.empty());

    double result = 0;
    switch (method) {
        case eHistogramComparisonChiSquare: {
            for (std::size_t j = 0 ; j < h1.size(); ++j) {
                double a = h1[j] - h2[j];
                if (std::fabs(h1[j] + h2[j]) > std::numeric_limits<double>::epsilon()) {
                    result += ((a * a) / (h1[j] + h2[j]));
                }
            }
        }   break;

        case eHistogramComparisonCorrelation: {
            double s1 = 0, s2 = 0, s11 = 0, s12 = 0, s22 = 0;

            for (std::size_t j = 0 ; j < h1.size(); ++j) {
                double a = h1[j];
                double b = h2[j];
                s12 += a*b;
                s1 += a;
                s11 += a*a;
                s2 += b;
                s22 += b*b;
            }
            double scale = 1. / h1.size();
            double num = s12 - s1*s2*scale;
            double denom2 = (s11 - s1*s1*scale)*(s22 - s2*s2*scale);
            result = std::abs(denom2) > std::numeric_limits<double>::epsilon() ? num/std::sqrt(denom2) : 1.;
        }   break;
        case eHistogramComparisonIntersection: {
            for (std::size_t j = 0 ; j < h1.size(); ++j) {
                result += std::min(h1[j], h2[j]);
            }
        }   break;
        case eHistogramComparisonBhattacharyya: {
            double s1 = 0, s2 = 0;

            for (std::size_t j = 0 ; j < h1.size(); ++j) {
                double a = h1[j];
                double b = h2[j];
                result += std::sqrt(a*b);
                s1 += a;
                s2 += b;
            }
            s1 *= s2;
            s1 = std::fabs(s1) > std::numeric_limits<double>::epsilon() ? 1./std::sqrt(s1) : 1.;
            result = std::sqrt(std::max(1. - result*s1, 0.));
        }   break;
        case eHistogramComparisonKLDiv: {
            for (std::size_t j = 0 ; j < h1.size(); ++j) {
                double p = h1[j];
                double q = h2[j];
                if( std::fabs(p) <= std::numeric_limits<double>::epsilon() ) {
                    continue;
                }
                if(  fabs(q) <= std::numeric_limits<double>::epsilon() ) {
                    q = 1e-10;
                }
                result += p * std::log( p / q );
            }
        }   break;
    } // switch (method)
    return result;
} // compareHist

inline void normalizeHistogram(std::vector<double>* histogram) {
    double s1 = 0;
    for (std::size_t i = 0; i < histogram->size(); ++i) {
        s1 += (*histogram)[i];
    }
    if (s1 == 0) {
        return;
    }

    for (std::size_t i = 0; i < histogram->size(); ++i) {
        (*histogram)[i] /= s1;
    }

}

inline void loadHistogramFromFile(const boost::shared_ptr<FStreamsSupport::ifstream>& file, int modelIndexInFile, const std::vector<int>& modelSizes, std::vector<double>* outModel)
{

    // Each model of detection is this amount of bytes in the model file

    std::size_t modelNumDouble = 1;
    for (std::size_t i = 0; i < modelSizes.size(); ++i) {
        modelNumDouble *= modelSizes[i];
    }
    outModel->resize(modelNumDouble);

    file->seekg(modelIndexInFile * modelNumDouble * sizeof(double), file->beg);
    file->read((char*)&(*outModel)[0], modelNumDouble * sizeof(double));
}


template <typename T>
class RamBuffer
{
    T* data;
    std::size_t count;

public:

    RamBuffer()
    : data(0)
    , count(0)
    {
    }

    ~RamBuffer()
    {
        clear();
    }

    T* getData()
    {
        return data;
    }

    const T* getData() const
    {
        return data;
    }

    void swap(RamBuffer& other)
    {
        std::swap(data, other.data);
        std::swap(count, other.count);
    }

    std::size_t size() const
    {
        return count;
    }

    void resize(std::size_t size)
    {
        if (size == 0) {
            return;
        }
        count = size;
        if (data) {
            free(data);
            data = 0;
        }
        if (count == 0) {
            return;
        }
        data = (T*)malloc( size * sizeof(T) );
        if (!data) {
            throw std::bad_alloc();
        }
    }

    void resizeAndPreserve(std::size_t size)
    {
        if (size == 0 || size == count) {
            return;
        }
        count = size;
        data = (T*)realloc(data,size * sizeof(T));
        if (!data) {
            throw std::bad_alloc();
        }
    }

    void clear()
    {
        count = 0;
        if (data) {
            free(data);
            data = 0;
        }
    }


};

/**
 * @brief Computes H and S histograms for the given rectangle
 **/
inline void computeHistograms(const RectI& roi, const image& image, std::vector<double>* histogramOut)
{
    RectI imageBounds;
    imageBounds.x1 = 0;
    imageBounds.y1 = 0;
    imageBounds.x2 = image.w;
    imageBounds.y2 = image.h;
    assert(imageBounds.contains(roi));


    RamBuffer<unsigned char> hsvImage;
    hsvImage.resize(roi.width() * roi.height() * image.c);

    convertRGBToHSV(image.c, eBitDepthFloat, image.data, imageBounds, roi, eBitDepthByte, hsvImage.getData(), roi);

    std::vector<int> histogramSizes(3);
    histogramSizes[0] = HUE_HISTOGRAM_NUM_BINS;
    histogramSizes[1] = SAT_HISTOGRAM_NUM_BINS;
    histogramSizes[2] = 0;

    computeHistogram(image.c, eBitDepthByte, hsvImage.getData(), roi, roi, histogramSizes, histogramOut);

    normalizeHistogram(histogramOut);


#if 0


    // Convert the image to a cv::Mat
    cv::Mat rgbMat(roi.height(), roi.width(), CV_8UC3);
    cv::Mat hsvMat;

    {
        uchar* dstPixels = rgbMat.data;
        assert(rgbMat.step.p[1] == 3);

        float* srcPixels = image.data + roi.y1 * image.w * image.c + roi.x1 * image.c;

        //return data + i0 * step.p[0] + i1 * step.p[1];
        for (int y = roi.y1; y < roi.y2; ++y) {
            for (int x = roi.x1; x < roi.x2; ++x,dstPixels += 3, srcPixels += image.c) {
                for (int c = 0; c < 3; ++c) {
                    float srcVal;
                    if (c < image.c) {
                        srcVal = srcPixels[c];
                    } else {
                        srcVal = 0;
                    }
                    dstPixels[c] = std::min(1.f,std::max(0.f, srcVal)) * 255.f;
                }
            }
            // Remove what was done on last iteration and go to the next line
            srcPixels += (image.w * image.c - roi.width() * image.c);
            dstPixels += (rgbMat.step.p[0] - roi.width() * 3);
        }
    }


    // Convert to HSV
    cv::cvtColor( rgbMat, hsvMat, cv::COLOR_BGR2HSV);

    // Using 32 bins for hue and 32 for saturation
    int histSize[] = { HUE_HISTOGRAM_NUM_BINS, SAT_HISTOGRAM_NUM_BINS };

    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };

    // Use the 0-th and 1-st channels
    int channels[] = { 0, 1 };

    cv::Mat hist;

    // Calculate the histograms for the HSV images
    calcHist( &hsvMat, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    assert(hist.type() ==  CV_32F);
    // normalize between 0 and 1
    cv::normalize( hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );


    histogramOut->resize(hist.rows * hist.cols);

    double* histData = &(*histogramOut)[0];
    for (int i = 0; i < hist.rows; ++i) {
        for (int j = 0; j < hist.cols; ++j) {
            *histData = hist.at<float>(i, j);
            ++histData;
        }
    }
#endif

    // Write in output
}


inline std::list<std::string>::iterator hasToken(std::list<std::string> &localArgs, const std::string& token)
{
    for (std::list<std::string>::iterator it = localArgs.begin(); it!=localArgs.end(); ++it) {
        if (*it == token) {
            return it;
        }
    }
    return localArgs.end();
} // hasToken

inline void split(const std::string &s, char delim, std::vector<std::string>* result) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        result->push_back(item);
    }
}


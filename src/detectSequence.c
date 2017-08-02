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

using std::string;

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

struct FetcherThreadArgs
{
    image inputImage, inputImageScaled;
    string filename;
    CvCapture* capture;
    network* net;
};

image get_image_from_stream(CvCapture *cap);
image ipl_to_image(IplImage* src);

void *fetch_in_thread(void *ptr)
{
    FetcherThreadArgs* args = (FetcherThreadArgs*)ptr;
    if (args->capture) {
        args->inputImage = get_image_from_stream(args->capture);
    } else {
        args->inputImage = load_image_color(const_cast<char*>(args->filename.c_str()),0,0);
    }
    args->inputImageScaled = resize_image(args->inputImage, args->net->w, args->net->h);
    return 0;
}

struct DetectThreadArgs
{
    // Changes for each detection
    int frameNumber;
    image inputImage, inputImageScaled;

    // Whether to write the file in output or not
    bool writeOutput;

    // Return code
    bool ok;

    // Never changes
    network* net;
    float* predictions;
    float** probs;
    float* avg;
    box *boxes;
    float thresh;
    float hierThresh;
    std::vector<string>* names;
    std::vector<string>* allowedClasses;
    FStreamsSupport::ofstream* modelFile;
    int modelFileIndex;
    int modelIndexInFile;
    string *outputFilename;
    SERIALIZATION_NAMESPACE::SequenceSerialization* outSeq;

};

template <typename PIX>
const PIX* getPixels(int x, int y, const RectI& bounds, int nComps, const PIX* pixelData)
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
PIX* getPixels(int x, int y, const RectI& bounds, int nComps, PIX* pixelData)
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
static DSTPIX convertPixelDepth(SRCPIX pix);


/// numvals should be 256 for byte, 65536 for 16-bits, etc.

/// maps 0-(numvals-1) to 0.-1.
template<int numvals>
float
intToFloat(int value)
{
    return value / (float)(numvals - 1);
}

/// maps 0.-1. to 0-(numvals-1)
template<int numvals>
int
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
void
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
void convertRGBToHSVInternal(const SRCPIX* srcPixelsData, const RectI& srcBounds, const RectI& roi, DSTPIX* dstPixelsData, const RectI& dstBounds)
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
void convertRGBToHSVForNComps(const SRCPIX* srcPixelsData, const RectI& srcBounds, const RectI& roi, BitDepthEnum dstDepth, void* dstPixelsData, const RectI& dstBounds)
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
void convertRGBToHSVForSrcBitDepth(int nComps, const SRCPIX* srcPixelsData, const RectI& srcBounds, const RectI& roi, BitDepthEnum dstDepth, void* dstPixelsData, const RectI& dstBounds)
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

void convertRGBToHSV(int nComps, BitDepthEnum srcDepth, const void* srcPixelsData, const RectI& srcBounds, const RectI& roi, BitDepthEnum dstDepth, void* dstPixelsData, const RectI& dstBounds)
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
void computeHistogramInternal(const PIX* srcPixelsData, const RectI& srcBounds, const RectI& roi, const std::vector<int>& histogramSizes, std::vector<double>* histogram)
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
void computeHistogramForNComps(int nComps, const PIX* srcPixelsData, const RectI& srcBounds, const RectI& roi, const std::vector<int>& histogramSizes, std::vector<double>* histogram)
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

void computeHistogram(int nComps, BitDepthEnum depth, const void* srcPixelsData, const RectI& srcBounds, const RectI& roi, const std::vector<int>& histogramSizes, std::vector<double>* histogram)
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


static void normalizeHistogram(std::vector<double>* histogram) {
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
static void computeHistograms(const RectI& roi, const image& image, std::vector<double>* histogramOut)
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

void *detect_in_thread(void *ptr)
{

    DetectThreadArgs* args = (DetectThreadArgs*)ptr;
    args->ok = true;

    float nms = .4;

    layer l = args->net->layers[args->net->n-1];
    float *X = args->inputImageScaled.data;
    float *prediction = network_predict(*args->net, X);

    memcpy(args->predictions, prediction, l.outputs*sizeof(float));

    mean_arrays(&args->predictions, 1, l.outputs, args->avg);
    l.output = args->avg;

    free_image(args->inputImageScaled);

    if (l.type == DETECTION) {
        get_detection_boxes(l, 1, 1, args->thresh, args->probs, args->boxes, 0);
    } else if (l.type == REGION) {
        get_region_boxes(l, 1, 1, args->thresh, args->probs, args->boxes, 0, 0, args->hierThresh);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) {
        do_nms(args->boxes, args->probs, l.w*l.h*l.n, l.classes, nms);
    }
     
    int gridSize = l.w * l.h * l.n;
    int nClasses = (int)args->names->size();

    int imgWidth = args->inputImage.w;
    int imgHeight = args->inputImage.h;


    SERIALIZATION_NAMESPACE::FrameSerialization frameSerialization;

    for (int i = 0; i < gridSize; ++i) {
        int class1 = max_index(args->probs[i], nClasses);
        float prob = args->probs[i][class1];
        if (prob <= args->thresh) {
            continue;
        }

        std::string& label = (*args->names)[class1];

        bool keepDetection = false;
        if (args->allowedClasses->empty()) {
            keepDetection = true;
        } else {
            for (std::size_t c = 0; c < args->allowedClasses->size(); ++c) {
                if ((*args->allowedClasses)[c] == label) {
                    keepDetection = true;
                    break;
                }
            }
        }
        if (!keepDetection) {
            continue;
        }

        SERIALIZATION_NAMESPACE::DetectionSerialization detection;
        detection.label = label;
        detection.score = std::max(0.f,std::min(1.f, prob));


        const box& b = args->boxes[i];

        detection.x1 = (b.x-b.w/2.) * imgWidth;
        detection.y1 = (b.y+b.h/2.) * imgHeight;
        detection.y1 = imgHeight - 1 - detection.y1;
        detection.x2 = (b.x+b.w/2.) * imgWidth + 1;
        detection.y2 = (b.y-b.h/2.) * imgHeight;
        detection.y2 = imgHeight - detection.y2;

        // Clamp
        detection.x1 = std::max(0.,std::min((double)imgWidth, detection.x1));
        detection.y1 = std::max(0.,std::min((double)imgHeight, detection.y1));
        detection.x2 = std::max(0.,std::min((double)imgWidth, detection.x2));
        detection.y2 = std::max(0.,std::min((double)imgHeight, detection.y2));
        
#ifdef OPENCV
        RectI roi;
        roi.x1 = std::floor(detection.x1);
        roi.y1 = std::floor(detection.y1);
        roi.x2 = std::ceil(detection.x2);
        roi.y2 = std::ceil(detection.y2);

        std::vector<double> histogramOut;
        computeHistograms(roi, args->inputImage, &histogramOut);

        const std::size_t dataSize = sizeof(double) * histogramOut.size();
        args->modelFile->write((const char*)&histogramOut[0], dataSize);

        detection.modelIndex = args->modelIndexInFile;
        ++args->modelIndexInFile;
        detection.fileIndex = args->modelFileIndex;
#endif

        frameSerialization.detections.push_back(detection);

    }

    free_image(args->inputImage);


    // No detection, do not write to the file
    if (frameSerialization.detections.empty()) {
        std::cerr << "No detection at frame " << args->frameNumber << std::endl;
    } else {
        args->outSeq->frames.insert(std::make_pair(args->frameNumber, frameSerialization));
    }


    // Write the results file
    if (args->writeOutput) {
        FStreamsSupport::ofstream ofile;
        FStreamsSupport::open(&ofile, *args->outputFilename);
        if (!ofile) {
            args->ok = false;
            return 0;
        }
        try {
            SERIALIZATION_NAMESPACE::write(ofile, *args->outSeq, SERIALIZATION_FILE_FORMAT_HEADER);
        } catch (const std::exception& e) {
            args->ok = false;
            fprintf(stderr, "%s", e.what());
            return 0;
        }
    }

    return 0;
} // detect_in_thread

static std::list<string>::iterator hasToken(std::list<string> &localArgs, const string& token)
{
    for (std::list<string>::iterator it = localArgs.begin(); it!=localArgs.end(); ++it) {
        if (*it == token) {
            return it;
        }
    }
    return localArgs.end();
} // hasToken

void split(const std::string &s, char delim, std::vector<std::string>* result) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        result->push_back(item);
    }
}

static void printUsage(const char* argv0)
{
    /* Text must hold in 80 columns ************************************************/
    std::stringstream ss;
    ss << "Usage:\n"
    << argv0
    <<
    " -i <inputVideo> -o <outputDetectionFile> --cfg <yoloConfigFile> --weights <yoloWeightsFile> [--names <classNamesFile>] [--range <firstFrame-lastFrame>] [--threshold <value>]\n"
    ;
    std::cout << ss.str() << std::endl;

}

static bool isVideoFile(const std::string& ext)
{
    std::string lower;
    for (std::size_t i = 0; i < ext.size(); ++i) {
        lower.push_back(std::tolower(ext[i]));
    }
    return lower == "mp4" || lower == "mov" || lower == "avi";
}

static void parseArguments(const std::list<string>& args,
                            std::vector<std::pair<std::string,int> > * inputFilesInOrder,
                           std::string *firstFrameFileName,
                           int* firstFrame, int* lastFrame,
                           CvCapture** videoStream,
                           int* numFrames,
                           string* cfgFilename,
                           string* weightFilename,
                           float* thresh,
                           std::vector<string>* names,
                           string* outputFilename)
{
    std::list<string> localArgs = args;

    SequenceParsing::SequenceFromPattern inputSequence;
    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "-i");
        if (foundInput == localArgs.end()) {
            throw std::invalid_argument("An input file must be specified with the -i switch.");
        }
        foundInput = localArgs.erase(foundInput);
        if ( foundInput == localArgs.end() ) {
            throw std::invalid_argument("-i switch without a file name");
        } else {
            SequenceParsing::filesListFromPattern_slow(*foundInput, &inputSequence);
            localArgs.erase(foundInput);
            if (inputSequence.empty()) {
                 throw std::invalid_argument("Empty input file");
            }
        }
    }

    bool isVideo;
    {
        *firstFrameFileName = inputSequence.begin()->second.begin()->second;
        std::string ext;
        std::size_t foundDot = firstFrameFileName->find_last_of(".");
        if (foundDot != std::string::npos) {
            ext = firstFrameFileName->substr(foundDot + 1);
        }
        isVideo = isVideoFile(ext);
    }


    int defaultFirstFrame,defaultLastFrame;
    if (!isVideo) {
        defaultFirstFrame = inputSequence.begin()->first;
        defaultLastFrame = inputSequence.rbegin()->first;
        *numFrames = defaultLastFrame - defaultFirstFrame + 1;

    } else {
        *videoStream = cvCaptureFromFile(firstFrameFileName->c_str());
        if (!videoStream) {
            throw std::invalid_argument("Could not open " + *firstFrameFileName);
        }

        *numFrames = cvGetCaptureProperty(*videoStream, CV_CAP_PROP_FRAME_COUNT);

        defaultFirstFrame = 1;
        defaultLastFrame = *numFrames - 1;

    }
    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "--range");
        if (foundInput == localArgs.end()) {
            *firstFrame = defaultFirstFrame;
            *lastFrame = defaultLastFrame;
        } else {
            foundInput = localArgs.erase(foundInput);
            if ( foundInput == localArgs.end() ) {
                throw std::invalid_argument("--range switch without a frame range");
            } else {
                std::vector<string> rangeVec;
                split(*foundInput, '-', &rangeVec);
                if (rangeVec.size() != 2) {
                    throw std::invalid_argument("Frame range must be in the form firstFrame-lastFrame");
                }
                {
                    std::stringstream ss(rangeVec[0]);
                    ss >> *firstFrame;
                }
                {
                    std::stringstream ss(rangeVec[1]);
                    ss >> *lastFrame;
                }


                if (*firstFrame < defaultFirstFrame || *lastFrame > defaultLastFrame) {
                    printf("Frame range must be in the sequence range (%i-%i)",defaultFirstFrame, defaultLastFrame);
                    exit(1);
                }
                localArgs.erase(foundInput);
            }
        }
        
    }
    {
        SequenceParsing::SequenceFromPattern::iterator startIt = inputSequence.find(*firstFrame);
        SequenceParsing::SequenceFromPattern::iterator endIt = inputSequence.find(*lastFrame);
        if (startIt == inputSequence.end() || endIt == inputSequence.end()) {
            throw std::invalid_argument("Invalid frame range");
        }
        ++endIt;
        for (SequenceParsing::SequenceFromPattern::iterator it = startIt; it != endIt; ++it) {
            inputFilesInOrder->push_back(std::make_pair(it->second.begin()->second, it->first));
        }
        *numFrames = inputFilesInOrder->size();

        if (inputFilesInOrder->empty()) {
            throw std::invalid_argument("At least one frame must be detected");
        }
    }
    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "-o");
        if (foundInput == localArgs.end()) {
            throw std::invalid_argument("An output file must be specified with the -o switch.");
        }
        foundInput = localArgs.erase(foundInput);
        if ( foundInput == localArgs.end() ) {
            throw std::invalid_argument("-o switch without a file name");
        } else {
            *outputFilename = *foundInput;
            localArgs.erase(foundInput);
        }
    }
    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "--cfg");
        if (foundInput == localArgs.end()) {
            throw std::invalid_argument("A config file must be specified with the --cfg switch.");
        }
        foundInput = localArgs.erase(foundInput);
        if ( foundInput == localArgs.end() ) {
            throw std::invalid_argument("--cfg switch without a file name");
        } else {
            *cfgFilename = *foundInput;
            localArgs.erase(foundInput);
        }
    }
    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "--weights");
        if (foundInput == localArgs.end()) {
            throw std::invalid_argument("A config file must be specified with the --weights switch.");
        }
        foundInput = localArgs.erase(foundInput);
        if ( foundInput == localArgs.end() ) {
            throw std::invalid_argument("--weights switch without a file name");
        } else {
            *weightFilename = *foundInput;
            localArgs.erase(foundInput);
        }
    }
    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "--names");
        std::string namesFile;
        if (foundInput == localArgs.end()) {
            printf("--names switch not specified, using data/names.list instead");
            namesFile = "data/names.list";
        } else {
            foundInput = localArgs.erase(foundInput);
            if ( foundInput == localArgs.end() ) {
                throw std::invalid_argument("--names switch without a file name");
            } else {
                namesFile = *foundInput;
                localArgs.erase(foundInput);
            }
        }
        {
            std::ifstream ifile;
            ifile.open(namesFile);
            if (!ifile) {
                fprintf(stderr, "Couldn't open file: %s\n", namesFile.c_str());
                exit(1);
            }
            string lineBuf;
            while (std::getline(ifile, lineBuf)){
                names->push_back(lineBuf);
            }
        }

    }

    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "--thresh");
        if (foundInput == localArgs.end()) {
            *thresh = 0.25;
            printf("--thresh not specified, using %f", *thresh);
        } else {
            foundInput = localArgs.erase(foundInput);
            if ( foundInput == localArgs.end() ) {
                throw std::invalid_argument("--thresh switch without a value");
            } else {
                std::stringstream ss(*foundInput);
                ss >> *thresh;
                localArgs.erase(foundInput);
            }
        }
    }

} // parseArguments

struct CaptureHolder
{
    CvCapture* _cap;
    CaptureHolder(CvCapture* cap)
    : _cap(cap)
    {

    }

    ~CaptureHolder()
    {
        if (_cap) {
            cvReleaseCapture(&_cap);
        }
    }
};


int renderImageSequence_main(int argc, char** argv)
{
    std::list<string> arguments;
    for (int i = 0; i < argc; ++i) {
        arguments.push_back(string(argv[i]));
    }

    CvCapture * videoStream = 0;
    std::vector<std::pair<std::string,int> > inputFilesInOrder;
    std::string firstFrameFileName;
    int firstFrame, lastFrame;
    int numFrames;
    string cfgFilename, weightFilename;
    float thresh;
    std::vector<string> names;
    string outputFile;
    float hier_thresh = .5;
    std::vector<string> allowedNames;
    allowedNames.push_back("person");
    parseArguments(arguments, &inputFilesInOrder, &firstFrameFileName, &firstFrame, &lastFrame, &videoStream, &numFrames, &cfgFilename, &weightFilename, &thresh, &names, &outputFile);

    CaptureHolder captureHolder(videoStream);


    // vector of pair<filename, framenumber> >


    SERIALIZATION_NAMESPACE::SequenceSerialization detectionResults;

    detectionResults.histogramSizes.push_back(HUE_HISTOGRAM_NUM_BINS);
    detectionResults.histogramSizes.push_back(SAT_HISTOGRAM_NUM_BINS);


    network net = parse_network_cfg(const_cast<char*>(cfgFilename.c_str()));
    load_weights(&net, const_cast<char*>(weightFilename.c_str()));
    set_batch_network(&net, 1);
    
    srand(2222222);
    
    layer detectionLayer = net.layers[net.n-1];
    
    float* predictions = (float *) calloc(detectionLayer.outputs, sizeof(float));
    float* avg = (float *) calloc(detectionLayer.outputs, sizeof(float));

    int gridSize = detectionLayer.w * detectionLayer.h * detectionLayer.n;

    box* boxes = (box *)calloc(gridSize, sizeof(box));
    float** probs = (float **)calloc(gridSize, sizeof(float *));

    for (int j = 0; j < gridSize; ++j) {
        probs[j] = (float *)calloc(detectionLayer.classes, sizeof(float));
    }


    // Name the model file containing the histograms exactly like outputFile but ending with _model
    std::string modelFileName = outputFile;
    std::string modelAbsoluteFileName;
    {
        // Remove extension
        std::size_t foundLastDot = modelFileName.find_last_of(".");
        if (foundLastDot != std::string::npos) {
            modelFileName = modelFileName.substr(0, foundLastDot);
        }
        modelFileName += "_model";
        modelAbsoluteFileName = modelFileName;

        // Remove path
        std::size_t foundSlash = modelFileName.find_last_of("/");
        if (foundLastDot != std::string::npos) {
            modelFileName = modelFileName.substr(foundSlash + 1);
        }
    }



    FetcherThreadArgs fetchArgs;
    fetchArgs.net = &net;

    DetectThreadArgs detectArgs;
    detectArgs.net = &net;
    detectArgs.predictions = predictions;
    detectArgs.probs = probs;
    detectArgs.avg = avg;
    detectArgs.boxes = boxes;
    detectArgs.thresh = thresh;
    detectArgs.hierThresh = hier_thresh;
    detectArgs.names = &names;
    detectArgs.modelFileIndex = 0;
    detectArgs.outSeq = &detectionResults;
    detectArgs.outputFilename = &outputFile;
    detectArgs.allowedClasses = &allowedNames;

    std::auto_ptr<FStreamsSupport::ofstream> modelFile;

    int curFrame_i = 0;

    while (curFrame_i < numFrames) {

        int frameNumber = !videoStream ? inputFilesInOrder[curFrame_i].second : curFrame_i + 1;
        const std::string& filename = !videoStream ? inputFilesInOrder[curFrame_i].first : firstFrameFileName;
        fetchArgs.capture = videoStream;
        fetchArgs.filename = filename;
        fetch_in_thread(&fetchArgs);

        if (!fetchArgs.inputImage.data) {
            throw std::invalid_argument("Could not open " + filename);
        }

        if (videoStream) {
            // Respect user frame range
            if (curFrame_i < firstFrame - 1) {
                continue;
            }
            if (curFrame_i > lastFrame - 1) {
                break;
            }
        }

        if (!modelFile.get() || detectArgs.modelIndexInFile >= MAX_NUM_MODELS_PER_FILE) {
            if (modelFile.get()) {
                modelFile->flush();
                modelFile->close();
                ++detectArgs.modelFileIndex;
            }
            detectArgs.modelIndexInFile = 0;
            modelFile.reset(new FStreamsSupport::ofstream);

            std::string actualFileName;
            {
                std::stringstream ss;
                ss << modelAbsoluteFileName << detectArgs.modelFileIndex +1;
                actualFileName = ss.str();
            }
            std::string actualRelativeFileName;
            {
                std::stringstream ss;
                ss << modelFileName << detectArgs.modelFileIndex +1;
                actualRelativeFileName = ss.str();
            }
            FStreamsSupport::open(modelFile.get(), actualFileName, std::ios_base::out | std::ios_base::trunc);
            if (!modelFile->is_open()) {
                throw std::invalid_argument("Could not open " + actualFileName);
            }


            detectionResults.modelFiles.push_back(actualRelativeFileName);
            detectArgs.modelFile = modelFile.get();
        }

        // Write if we reach the last one
        detectArgs.writeOutput = (curFrame_i == numFrames - 1 || curFrame_i == lastFrame - 1);
        detectArgs.frameNumber = frameNumber;
        detectArgs.inputImage = fetchArgs.inputImage;
        detectArgs.inputImageScaled = fetchArgs.inputImageScaled;
        std::cout << "Running detector on frame " << frameNumber << std::endl;
        detect_in_thread(&detectArgs);
        if (!detectArgs.ok) {
            return 1;
        }

        ++curFrame_i;
    }
    if (modelFile.get()) {
        modelFile->flush();
        modelFile->close();
    }
    
    return 0;
} // renderImageSequence_main

int main(int argc, char** argv)
{
    try {
        return renderImageSequence_main(argc, argv);
    } catch (const std::exception& e) {
        fprintf(stderr, "%s", e.what());
        printUsage(argv[0]);
        exit(1);
    }
}

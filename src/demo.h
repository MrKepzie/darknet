#ifndef DEMO
#define DEMO

#include "image.h"
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, float hier_thresh);

void writeout(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, const char *outputFilenamePrefix, float hier_thresh);
#endif

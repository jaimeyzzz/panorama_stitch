#ifndef PANORAMASTITCH_PANOSCRIPT_H_
#define PANORAMASTITCH_PANOSCRIPT_H_

#include "pano_common.h"

#include <string>
#include <vector>

class PanoScript {
public:
    PanoScript(std::string);
    ~PanoScript();

    void LoadScript(const std::string&);
private:
    void ReadPanoLine(const std::string&);
    ImageParam ReadImageLine(const std::string&);
public:
    int num_images;
    PanoParam panorama;
    std::vector<ImageParam> images;
};

#endif // PANORAMASTITCH_PANOSCRIPT_H_
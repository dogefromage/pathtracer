#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "settings.h"
#include "config.h"
#include <vector>

template <typename T>
struct ParseRule {
    std::string section, key, hint;
    T* (*selector)(settings_t& s);
    T defaultValue;
};

enum class ParserMode {
    Parse,
    Print,
};

struct YamlParser {
    settings_t& settings;

    std::vector<ParseRule<int>> intRules;
    std::vector<ParseRule<float>> floatRules;
    std::vector<ParseRule<bool>> boolRules;
    std::vector<ParseRule<Vec3>> vec3Rules;

    ParserMode mode;
    std::string lastSection;

    YamlParser(settings_t& _settings) : settings(_settings) {}

    void printSection(const std::string& section) {
        if (section != lastSection) {
            std::cout << section << ":\n";
            lastSection = section;
        }
    }

    void addInt(ParseRule<int>&& r) {
        if (mode == ParserMode::Parse) {
            *r.selector(settings) = r.defaultValue;
            intRules.push_back(r);
        }
        if (mode == ParserMode::Print) {
            printSection(r.section);
            std::cout << "  " << r.key << ": " << *r.selector(settings) << std::endl;
        }
    }

    void addFloat(ParseRule<float>&& r) {
        if (mode == ParserMode::Parse) {
            *r.selector(settings) = r.defaultValue;
            floatRules.push_back(r);
        }
        if (mode == ParserMode::Print) {
            printSection(r.section);
            std::cout << "  " << r.key << ": " << *r.selector(settings) << std::endl;
        }
    }

    void addBool(ParseRule<bool>&& r) {
        if (mode == ParserMode::Parse) {
            *r.selector(settings) = r.defaultValue;
            boolRules.push_back(r);
        }
        if (mode == ParserMode::Print) {
            printSection(r.section);
            std::cout << "  " << r.key << ": " << *r.selector(settings) << std::endl;
        }
    }

    void addVec3(ParseRule<Vec3>&& r) {
        if (mode == ParserMode::Parse) {
            *r.selector(settings) = r.defaultValue;
            vec3Rules.push_back(r);
        }
        if (mode == ParserMode::Print) {
            printSection(r.section);
            std::cout << "  " << r.key << ": " << *r.selector(settings) << std::endl;
        }
    }

    void process_rule(settings_t& settings, const std::string& section, const std::string& key, const std::string& value) {

        for (const auto& r : intRules) {
            if (r.section == section && r.key == key) {
                *r.selector(settings) = atoi(value.c_str());
                return;
            }
        }
        for (const auto& r : floatRules) {
            if (r.section == section && r.key == key) {
                *r.selector(settings) = (float)atof(value.c_str());
                return;
            }
        }
        for (const auto& r : boolRules) {
            if (r.section == section && r.key == key) {
                *r.selector(settings) = (value == "true");
                return;
            }
        }
        for (const auto& r : vec3Rules) {
            if (r.section == section && r.key == key) {
                Vec3* v = r.selector(settings);
                if (sscanf(value.c_str(), "%f %f %f", &v->x, &v->y, &v->z) < 3) {
                    printf("Yaml parsing error, expected 3 floats in key value pair: %s, %s\n", 
                        key.c_str(), value.c_str());
                    exit(EXIT_FAILURE);
                }
                return;
            }
        }
        std::cerr << ("Unknown setting in yaml file \"" + (section + "." + key) + "\"") << std::endl;
    }

    void parse(settings_t& settings, const std::string& filename) {

        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Unable to open file: " << filename << std::endl;
            exit(EXIT_FAILURE);
        }

        std::string line;
        std::string currentSection;

        while (std::getline(file, line)) {
            line.erase(0, line.find_first_not_of(" \t"));
            line.erase(line.find_last_not_of(" \t") + 1);

            if (line.empty()) continue;

            if (line.back() == ':') {
                currentSection = line.substr(0, line.size() - 1);
            } else {
                std::istringstream iss(line);
                std::string key;
                if (std::getline(iss, key, ':')) {
                    std::string value;
                    std::getline(iss, value);

                    key.erase(0, key.find_first_not_of(" \t"));
                    key.erase(key.find_last_not_of(" \t") + 1);
                    value.erase(0, value.find_first_not_of(" \t"));
                    value.erase(value.find_last_not_of(" \t") + 1);

                    process_rule(settings, currentSection, key, value);
                }
            }
        }

        file.close();
    }
};

static void create_parser(YamlParser& parser, ParserMode mode) {
    parser.mode = mode;

    parser.addInt({ "output", "width", "Width of output image", 
        [](settings_t& s) { return &s.output.width; }, 600 });
    parser.addInt({ "output", "height", "Height of output image", 
        [](settings_t& s) { return &s.output.height; }, 600 });

    parser.addInt({ "sampling", "seed", "Random seed", 
        [](settings_t& s) { return &s.sampling.seed; }, 42 });
    parser.addInt({ "sampling", "samples", "Number of samples per pixel", 
        [](settings_t& s) { return &s.sampling.samples; }, 100 });
    parser.addInt({ "sampling", "samples_per_round", "Send image to screen every n samples", 
        [](settings_t& s) { return &s.sampling.samples_per_round; }, 10000 });

    parser.addVec3({ "world", "clear_color", "Color of skybox", 
        [](settings_t& s) { return &s.world.clear_color; }, Vec3::Zero() });
}

void settings_parse_yaml(settings_t& settings, const std::string& filename) {
    YamlParser parser(settings);
    create_parser(parser, ParserMode::Parse);
    parser.parse(settings, filename);
}

void settings_print(settings_t& settings) {

    YamlParser printingParser(settings);

    printf("----------------------------------------\n");
    std::cout << "Settings:\n";
    
    create_parser(printingParser, ParserMode::Print); // prints settings

    printf("----------------------------------------\n");
}


#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "logger.h"
#include "utils.h"

static int current_log_level = LOG_LEVEL_INFO;
static FILE *log_file = NULL;
static int log_to_stdout = 1;

static void log_message(int level, const char *prefix, const char *format, va_list args) {
    if (level > current_log_level) {
        return;
    }

    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);

    if (log_file) {
        fprintf(log_file, "[%s] %s", prefix, buffer);
        fflush(log_file);
    }

    if (log_to_stdout) {
        fprintf(stdout, "[%s] %s", prefix, buffer);
        fflush(stdout);
    }
}

int log_init(int level, char *path_logfile, int also_stdout) {
    current_log_level = level;
    log_to_stdout = also_stdout;
    if (!(log_file = fopen(path_logfile, "w"))) {
        fprintf(stderr, "unable to open logfile at: %s\n", path_logfile);
        return 1;
    }
    return 0;

    va_list args;
    log_message(-1, "INIT", "Initialized logger!", args);
}

void log_close() {
    if (log_file && log_file != stdout && log_file != stderr) {
        fclose(log_file);
        log_file = NULL;
    }
}

void log_trace(const char *format, ...) {
    va_list args;
    va_start(args, format);
    log_message(LOG_LEVEL_TRACE, "TRACE", format, args);
    va_end(args);
}

void log_info(const char *format, ...) {
    va_list args;
    va_start(args, format);
    log_message(LOG_LEVEL_INFO, "INFO", format, args);
    va_end(args);
}

void log_warning(const char *format, ...) {
    va_list args;
    va_start(args, format);
    log_message(LOG_LEVEL_WARNING, "WARNING", format, args);
    va_end(args);
}

void log_error(const char *format, ...) {
    va_list args;
    va_start(args, format);
    log_message(LOG_LEVEL_ERROR, "ERROR", format, args);
    print_stacktrace();
    va_end(args);
}

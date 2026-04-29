#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#include <limits.h>

// 2MB Buffer for fast writing
#define IO_BUFFER_SIZE (2 * 1024 * 1024)

// --- Helper Functions ---

static int ends_with(const char *s, const char *suffix) {
    size_t ls = strlen(s), lf = strlen(suffix);
    return (ls >= lf) && (memcmp(s + (ls - lf), suffix, lf) == 0);
}

static int is_regular_file(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return S_ISREG(st.st_mode);
}

// Sort files alphabetically so data is in order of run01, run02...
static int cmp_strptr(const void *a, const void *b) {
    return strcmp(*(const char * const *)a, *(const char * const *)b);
}

static int list_csv_files_sorted(const char *dirpath, char ***out_list, size_t *out_n) {
    DIR *d = opendir(dirpath);
    if (!d) return -1;

    size_t cap = 32, n = 0;
    char **list = (char**)malloc(cap * sizeof(char*));
    
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        if (!ends_with(ent->d_name, ".csv")) continue;

        char full[PATH_MAX];
        snprintf(full, sizeof(full), "%s/%s", dirpath, ent->d_name);
        
        if (!is_regular_file(full)) continue;

        if (n == cap) {
            cap *= 2;
            char **nl = (char**)realloc(list, cap * sizeof(char*));
            if (!nl) { closedir(d); return -1; }
            list = nl;
        }
        list[n] = strdup(full);
        n++;
    }
    closedir(d);

    if (n > 0) qsort(list, n, sizeof(char*), cmp_strptr);
    *out_list = list;
    *out_n = n;
    return 0;
}

static void free_list(char **list, size_t n) {
    for (size_t i = 0; i < n; i++) free(list[i]);
    free(list);
}

// --- Main Merge Logic ---

void merge_class(const char *input_dir, const char *output_file, int label) {
    char **files = NULL;
    size_t nfiles = 0;

    // 1. Get list of files
    if (list_csv_files_sorted(input_dir, &files, &nfiles) != 0 || nfiles == 0) {
        fprintf(stderr, "[WARN] Skipping %s (Folder not found or empty)\n", input_dir);
        if (nfiles > 0) free_list(files, nfiles);
        return;
    }

    FILE *out = fopen(output_file, "w");
    if (!out) {
        perror("Error creating output file");
        free_list(files, nfiles);
        return;
    }

    // Set large buffer
    char *out_buf = malloc(IO_BUFFER_SIZE);
    if (out_buf) setvbuf(out, out_buf, _IOFBF, IO_BUFFER_SIZE);

    printf("[INFO] Merging %zu files from '%s' -> '%s' (Label: %d)\n", nfiles, input_dir, output_file, label);

    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    int header_written = 0;

    for (size_t i = 0; i < nfiles; i++) {
        FILE *in = fopen(files[i], "r");
        if (!in) { perror(files[i]); continue; }

        int is_first_line = 1;

        while ((read = getline(&line, &len, in)) != -1) {
            // Trim newlines
            while (read > 0 && (line[read-1] == '\n' || line[read-1] == '\r')) {
                line[--read] = '\0';
            }

            if (is_first_line) {
                is_first_line = 0;
                // Only write the header once (for the very first file)
                if (!header_written) {
                    fprintf(out, "%s,target\n", line);
                    header_written = 1;
                }
                continue; 
            }

            // Write data line + label
            fprintf(out, "%s,%d\n", line, label);
        }
        fclose(in);
    }

    free(line);
    free_list(files, nfiles);
    fclose(out);
    if (out_buf) free(out_buf);
    
    printf("[SUCCESS] Created %s\n", output_file);
}

int main(int argc, char **argv) {
    // If user provides a path arg, use it. Otherwise assume "."
    const char *base = (argc >= 2) ? argv[1] : ".";
    
    char p1[PATH_MAX], p2[PATH_MAX], p3[PATH_MAX];
    char o1[PATH_MAX], o2[PATH_MAX], o3[PATH_MAX];

    // Define Input Paths
    snprintf(p1, sizeof(p1), "%s/pattern_1", base);
    snprintf(p2, sizeof(p2), "%s/pattern_2", base);
    snprintf(p3, sizeof(p3), "%s/pattern_3", base);

    // Define Output Paths (saving to the base directory)
    snprintf(o1, sizeof(o1), "l1_traces.csv");
    snprintf(o2, sizeof(o2), "l2_traces.csv");
    snprintf(o3, sizeof(o3), "dram_traces.csv");

    // Execute Merges
    // Label 0 = L1, Label 1 = L2, Label 2 = DRAM
    merge_class(p1, o1, 0);
    merge_class(p2, o2, 1);
    merge_class(p3, o3, 2);

    return 0;
}
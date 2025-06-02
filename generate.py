import os
import random
import subprocess
import multiprocessing
import shutil
import sys
import re
import argparse

TILE_DIM = 16  # Global TILE_DIM for P7, can be referenced by P7 test case generator

# Helper functions


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def write_file(path, content):
    with open(path, "w", encoding='utf-8') as f:
        f.write(content)

# --- Problem 1: Vector Addition ---


P1_TEST_CASES_HARDCODED = [
    ({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]}, "Typical Case"),
    ({"A": [100], "B": [200]}, "Single Element"),
    ({"A": [], "B": []}, "Empty Vectors"),
    ({"A": [-1, -2, 0], "B": [1, 2, 5]}, "Negative Numbers"),
    ({"A": [1000000000, 5], "B": [1000000000, 6]},
     "Large Numbers (no overflow for int sum)")
]


def p1_generate_random_case():
    N = random.randint(1, 2048)
    A = [random.randint(-1000, 1000) for _ in range(N)]
    B = [random.randint(-1000, 1000) for _ in range(N)]
    return {"A": A, "B": B}

# --- Problem 2: SAXPY ---


P2_TEST_CASES_HARDCODED = [
    ({"alpha": 2, "A": [1, 2, 3], "B": [10, 20, 30]}, "Typical Case"),
    ({"alpha": 0, "A": [1, 2, 3], "B": [10, 20, 30]}, "Alpha is Zero"),
    ({"alpha": 1, "A": [-1, 5], "B": [1, -5]}, "Alpha is One, results zero"),
    ({"alpha": 5, "A": [], "B": []}, "Empty Vectors"),
    ({"alpha": -1, "A": [1, 2], "B": [1, 2]}, "Negative Alpha, results zero")
]


def p2_generate_random_case():
    N = random.randint(1, 2048)
    alpha = random.randint(-10, 10)
    A = [random.randint(-100, 100) for _ in range(N)]
    B = [random.randint(-100, 100) for _ in range(N)]
    return {"alpha": alpha, "A": A, "B": B}

# --- Problem 3: 1D Stencil ---


P3_TEST_CASES_HARDCODED = [
    ({"A": [3, 6, 9, 12, 15]},
     "Typical Case. Expected: [(3+3+6)/3, (3+6+9)/3, (6+9+12)/3, (9+12+15)/3, (12+15+15)/3] = [4,6,9,12,14]"),
    ({"A": [9]}, "Single Element. Expected: [(9+9+9)/3] = [9]"),
    ({"A": [3, 9]}, "Two Elements. Expected: [(3+3+9)/3, (3+9+9)/3] = [5,7]"),
    ({"A": [7, 7, 7, 7]}, "All Same. Expected: [7,7,7,7]"),
    ({"A": []}, "Empty Vector. Expected: []")
]


def p3_generate_random_case():
    N = random.randint(1, 1024)
    A = [random.randint(0, 255) for _ in range(N)]  # Simulating pixel values
    return {"A": A}


# --- Problem 4: Matrix Transposition (Naive) ---
P4_TEST_CASES_HARDCODED = [
    ({"M": 2, "K": 3, "A_flat": [1, 2, 3, 4, 5, 6]}, "Typical 2x3 to 3x2"),
    ({"M": 2, "K": 2, "A_flat": [1, 2, 3, 4]}, "Square 2x2"),
    ({"M": 1, "K": 4, "A_flat": [10, 20, 30, 40]}, "Row Vector 1x4 to 4x1"),
    ({"M": 3, "K": 1, "A_flat": [5, 10, 15]}, "Col Vector 3x1 to 1x3"),
    ({"M": 0, "K": 5, "A_flat": []}, "Empty Matrix (0x5)"),
    ({"M": 3, "K": 0, "A_flat": []}, "Empty Matrix (3x0)"),
    ({"M": 0, "K": 0, "A_flat": []}, "Empty Matrix (0x0)")
]


def p4_generate_random_case():
    M = random.randint(1, 64)
    K = random.randint(1, 64)
    if random.random() < 0.1:
        M = random.randint(0, 1)  # Test small/zero dimensions
    if random.random() < 0.1:
        K = random.randint(0, 1)

    A_flat = [random.randint(-100, 100) for _ in range(M * K)]
    return {"M": M, "K": K, "A_flat": A_flat}


# --- Problem 5: Parallel Reduction (Sum) ---
P5_TEST_CASES_HARDCODED = [
    ({"A": [1, 2, 3, 4, 5, 6, 7, 8]}, "Typical Case (N=8)"),  # Sum = 36
    ({"A": [100]}, "Single Element"),  # Sum = 100
    ({"A": [0, 0, 0, 0]}, "All Zeros"),  # Sum = 0
    ({"A": [-1, -2, -3, 10]}, "Negative Numbers"),  # Sum = 4
    ({"A": []}, "Empty Vector")  # Sum = 0
]


def p5_generate_random_case():
    N = random.randint(1, 4096)  # Can be larger for reduction
    A = [random.randint(-100, 100) for _ in range(N)]
    return {"A": A}


# --- Problem 6: 2D Convolution ---
# --- Problem 6: 2D Convolution ---
P6_TEST_CASES_HARDCODED = [
    ({"H": 3, "W": 3, "Img": [10, 20, 30, 40, 50, 60, 70, 80, 90], 
      "KH": 3, "KW": 3, "Kernel": [0, -1, 0, -1, 5, -1, 0, -1, 0]}, "3x3 Image with 3x3 Kernel"),
    ({"H": 4, "W": 4, "Img": [0, 0, 0, 0, 0, 18, 18, 0, 0, 18, 18, 0, 0, 0, 0, 0], 
      "KH": 3, "KW": 3, "Kernel": [1, 0, -1, 2, 0, -2, 1, 0, -1]}, "4x4 Image with 3x3 Edge Detection Kernel"),
    ({"H": 1, "W": 5, "Img": [10, 20, 30, 40, 50], 
      "KH": 1, "KW": 3, "Kernel": [0.25, 0.5, 0.25]}, "1x5 Row Image with 1x3 Blur Kernel"),
    ({"H": 5, "W": 1, "Img": [10, 20, 30, 40, 50], 
      "KH": 3, "KW": 1, "Kernel": [0.25, 0.5, 0.25]}, "5x1 Col Image with 3x1 Blur Kernel"),
    ({"H": 0, "W": 0, "Img": [], 
      "KH": 3, "KW": 3, "Kernel": [0, 0, 0, 0, 1, 0, 0, 0, 0]}, "0x0 Empty Image with 3x3 Identity Kernel"),
]


def p6_generate_random_case():
    H = random.randint(1, 64)
    W = random.randint(1, 64)
    if random.random() < 0.1:
        H = random.randint(0, 3)
    if random.random() < 0.1:
        W = random.randint(0, 3)
    
    KH = random.choice([1, 3, 5, 7])
    KW = random.choice([1, 3, 5, 7])
    Img = [random.randint(0, 255) for _ in range(H * W)]
    Kernel = [random.uniform(-2.0, 2.0) for _ in range(KH * KW)]
    
    return {"H": H, "W": W, "Img": Img, "KH": KH, "KW": KW, "Kernel": Kernel}


# --- Problem 7: Tiled Matrix Multiplication ---
P7_TEST_CASES_HARDCODED = [
    ({"M": 2, "K": 2, "N": 2, "A": [1, 2, 3, 4], "B": [
     5, 6, 7, 8]}, "2x2 * 2x2"),  # C = [[19,22],[43,50]]
    ({"M": 1, "K": 3, "N": 1, "A": [1, 2, 3], "B": [
     4, 5, 6]}, "1x3 * 3x1 (Dot product)"),  # C = [[32]]
    ({"M": 2, "K": 1, "N": 2, "A": [1, 2], "B": [
     3, 4]}, "2x1 * 1x2 (Outer product)"),  # C = [[3,4],[6,8]]
    ({"M": TILE_DIM, "K": TILE_DIM, "N": TILE_DIM, "A": [1.0]*(TILE_DIM*TILE_DIM), "B": [
     2.0]*(TILE_DIM*TILE_DIM)}, f"{TILE_DIM}x{TILE_DIM} ones * {TILE_DIM}x{TILE_DIM} twos"),
    ({"M": 0, "K": 5, "N": 5, "A": [], "B": [1.0]*25}, "M=0"),
    # C should be 5x5 of zeros
    ({"M": 5, "K": 0, "N": 5, "A": [], "B": []}, "K=0"),
    ({"M": 5, "K": 5, "N": 0, "A": [1.0]*25, "B": []}, "N=0"),
]


def p7_generate_random_case():
    m_factor = random.randint(1, 3)
    k_factor = random.randint(1, 3)
    n_factor = random.randint(1, 3)
    # Make it not always a multiple of TILE_DIM
    M = TILE_DIM * m_factor - random.randint(0, TILE_DIM//2)
    K = TILE_DIM * k_factor - random.randint(0, TILE_DIM//2)
    N = TILE_DIM * n_factor - random.randint(0, TILE_DIM//2)
    M = max(0, M)
    K = max(0, K)
    N = max(0, N)  # Ensure non-negative
    if random.random() < 0.1:
        M = 0
    elif random.random() < 0.1:
        K = 0
    elif random.random() < 0.1:
        N = 0

    A = [random.uniform(-5.0, 5.0) for _ in range(M * K)]
    B = [random.uniform(-5.0, 5.0) for _ in range(K * N)]
    return {"M": M, "K": K, "N": N, "A": A, "B": B}


# --- Problem 8: Parallel Histogram ---
NUM_BINS_P8 = 256  # For problem 8, histogram bins

P8_TEST_CASES_HARDCODED = [
    ({"Data": [0, 1, 2, 0, 1, 0, 5, 10, NUM_BINS_P8-1]}, "Simple small data"),
    ({"Data": [i % NUM_BINS_P8 for i in range(1000)]},
     "Uniform distribution up to 1000"),
    ({"Data": [0]*500 + [NUM_BINS_P8-1]*500}, "Two bins heavily populated"),
    ({"Data": []}, "Empty data"),
    ({"Data": [10]*(NUM_BINS_P8*2)},
     f"Single value repeated {NUM_BINS_P8*2} times"),
]


def p8_generate_random_case():
    N = random.randint(1, 8192)
    Data = [random.randint(0, NUM_BINS_P8 - 1) for _ in range(N)]
    if random.random() < 0.05:
        N = 0
        Data = []
    return {"Data": Data}

# --- Problem 9: K-Means Clustering ---
P9_TEST_CASES_HARDCODED = [
    ({"N": 5, "K": 2, "iter": 10, "points": [
     (0, 0), (1, 1), (10, 10), (11, 11), (5, 5)]}, "Simple 5 points, 2 clusters"),
    ({"N": 100, "K": 3, "iter": 20, "points": [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(30)] +
      [(random.uniform(5, 6), random.uniform(5, 6)) for _ in range(30)] +
      [(random.uniform(10, 11), random.uniform(0, 1)) for _ in range(40)]}, "100 points, 3 clear clusters"),
    ({"N": 10, "K": 10, "iter": 5, "points": [
     (i, i) for i in range(10)]}, "N=K, each point its own cluster"),
    ({"N": 20, "K": 1, "iter": 5, "points": [
     (random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(20)]}, "K=1, all points one cluster"),
    ({"N": 0, "K": 3, "iter": 10, "points": []}, "Zero points"),
    ({"N": 10, "K": 0, "iter": 10, "points": [
     (i, i) for i in range(10)]}, "Zero clusters"),
]


def p9_generate_random_case():
    N = random.randint(0, 512)
    K = random.randint(0, 10)
    if N == 0:
        K = 0
    elif K > N:
        K = N

    max_iter = random.randint(5, 30)
    points = []
    # Generate K "centers" for clusters first
    cluster_centers = []
    for _ in range(K):
        cluster_centers.append(
            (random.uniform(-10, 10), random.uniform(-10, 10)))

    for i in range(N):
        if K > 0:
            center = cluster_centers[i % K]  # Assign point to a cluster center
            points.append((random.normalvariate(
                center[0], 2.0), random.normalvariate(center[1], 2.0)))
        else:  # No clusters, just random points
            points.append((random.uniform(-10, 10), random.uniform(-10, 10)))

    return {"N": N, "K": K, "iter": max_iter, "points": points}



def get_makefile_content(problem_name_short, test_prefixes, output_file_template_student, config, gpu_arch_val):

    output_file_template_expected = output_file_template_student.replace(
        "_stud_", "_expected_")
    test_cmd_args_template = config['student_app_args_template_make']

    test_cases_str = " ".join(test_prefixes)
    first_test_case = test_prefixes[0]

    return f"""NVCC = nvcc
# Your GPU's compute capability
GPU_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
ARCH := sm_$(shell echo $(GPU_CC) | tr -d '.')
NVCC_FLAGS = -std=c++11 -arch=$(ARCH)  -Xcompiler -Wno-unused-function --expt-relaxed-constexpr
# For debugging, uncomment next line and comment line above
# NVCC_FLAGS = -std=c++11 -arch=$(ARCH) -G -g -Xcompiler -Wall -Xcompiler -Wno-unused-function

TARGET_STUDENT = student_exec
TARGET_ANSWER = answer_exec
SRC_STUDENT = student.cu
SRC_ANSWER = answer.cu

TEST_CASES = {test_cases_str}
FIRST_TEST_CASE = {first_test_case}

.PHONY: all student answer test clean run

all: student

student: $(TARGET_STUDENT)

answer: $(TARGET_ANSWER)

$(TARGET_STUDENT): $(SRC_STUDENT)
	@echo "Compiling student solution for {problem_name_short}..."
	$(NVCC) $(NVCC_FLAGS) $(SRC_STUDENT) -o $(TARGET_STUDENT)
	@echo "Student solution compiled as $(TARGET_STUDENT)."

$(TARGET_ANSWER): $(SRC_ANSWER)
	@echo "Compiling reference answer for {problem_name_short}..."
	$(NVCC) $(NVCC_FLAGS) $(SRC_ANSWER) -o $(TARGET_ANSWER)
	@echo "Reference answer compiled as $(TARGET_ANSWER)."

test: $(TARGET_STUDENT) $(TARGET_ANSWER) 
	@echo "Running tests for {problem_name_short}..."
	@ALL_PASSED=true; \\
	for test_case in $(TEST_CASES); do \\
		echo ""; \\
		echo "--- Running test case: $$test_case ---"; \\
		STUD_OUTPUT_FILE="{output_file_template_student.format(tc='$${test_case}')}"; \\
		EXPECT_OUTPUT_FILE="{output_file_template_expected.format(tc='$${test_case}')}"; \\
		\\
		if [ ! -f "$$EXPECT_OUTPUT_FILE" ]; then \\
		    echo "Expected output file $$EXPECT_OUTPUT_FILE not found. Generating it now using answer_exec..."; \\
		    ./$(TARGET_ANSWER) {config['app_args_template_make'].format(tc='$${test_case}')} test_data/`basename $$EXPECT_OUTPUT_FILE`; \\
            if [ ! -f "$$EXPECT_OUTPUT_FILE" ]; then \\
                echo "ERROR: Failed to generate expected output file $$EXPECT_OUTPUT_FILE."; \\
                ALL_PASSED=false; \\
                continue; \\
            fi; \\
		fi; \\
		\\
		./$(TARGET_STUDENT) {test_cmd_args_template.format(tc='$${test_case}')}; \\
		\\
		if [ ! -f "$$STUD_OUTPUT_FILE" ]; then \\
		    echo "Test $$test_case FAILED: Student output file '$$STUD_OUTPUT_FILE' not found."; \\
		    ALL_PASSED=false; \\
		elif cmp -s "$$STUD_OUTPUT_FILE" "$$EXPECT_OUTPUT_FILE"; then \\
			echo "Test $$test_case PASSED"; \\
		else \\
			echo "Test $$test_case FAILED"; \\
			echo "------------------------- STUDENT OUTPUT (differs) ------------------------"; \\
			head -n 20 "$$STUD_OUTPUT_FILE"; \\
            if [ `cat "$$STUD_OUTPUT_FILE" | wc -l` -gt 20 ]; then echo "... (output truncated)"; fi; \\
			echo ""; \\
			echo "------------------------- EXPECTED OUTPUT -----------------------"; \\
			head -n 20 "$$EXPECT_OUTPUT_FILE"; \\
            if [ `cat "$$EXPECT_OUTPUT_FILE" | wc -l` -gt 20 ]; then echo "... (output truncated)"; fi; \\
			echo "-----------------------------------------------------------------"; \\
			ALL_PASSED=false; \\
		fi; \\
	done; \\
	echo ""; \\
	if $$ALL_PASSED; then \\
		echo "********************* All tests for {problem_name_short} passed! *********************"; \\
		exit 0; \\
	else \\
		echo "!!!!!!!!!!!!!!!!!!!!! Some tests for {problem_name_short} failed. !!!!!!!!!!!!!!!!!!!!!"; \\
		exit 1; \\
	fi

run: $(TARGET_STUDENT) $(TARGET_ANSWER)
	@if [ -z "$(TC)" ]; then \\
		echo "Usage: make run TC=<test_case_prefix>"; \\
		exit 1; \\
	fi
	@echo ""; \\
	echo "--- Running single test case: $(TC) ---"; \\
	STUD_OUTPUT_FILE="{output_file_template_student.format(tc='$(TC)')}"; \\
	EXPECT_OUTPUT_FILE="{output_file_template_expected.format(tc='$(TC)')}"; \\
	\\
	if [ ! -f "$$EXPECT_OUTPUT_FILE" ]; then \\
		echo "Expected output file $$EXPECT_OUTPUT_FILE not found. Generating it now using answer_exec..."; \\
		./$(TARGET_ANSWER) {config['app_args_template_make'].format(tc='$(TC)')} test_data/`basename $$EXPECT_OUTPUT_FILE`; \\
		if [ ! -f "$$EXPECT_OUTPUT_FILE" ]; then \\
			echo "ERROR: Failed to generate expected output file $$EXPECT_OUTPUT_FILE."; \\
			exit 1; \\
		fi; \\
	fi; \\
	\\
	./$(TARGET_STUDENT) {test_cmd_args_template.format(tc='$(TC)')}; \\
	\\
	if [ ! -f "$$STUD_OUTPUT_FILE" ]; then \\
		echo "Test $(TC) FAILED: Student output file '$$STUD_OUTPUT_FILE' not found."; \\
		exit 1; \\
	elif cmp -s "$$STUD_OUTPUT_FILE" "$$EXPECT_OUTPUT_FILE"; then \\
		echo "Test $(TC) PASSED"; \\
		exit 0; \\
	else \\
		echo "Test $(TC) FAILED"; \\
		echo "------------------------- STUDENT OUTPUT (differs) ------------------------"; \\
		cat "$$STUD_OUTPUT_FILE"; \\
		echo ""; \\
		echo "------------------------- EXPECTED OUTPUT -----------------------"; \\
		cat "$$EXPECT_OUTPUT_FILE"; \\
		echo "-----------------------------------------------------------------"; \\
		exit 1; \\
	fi

list:
	@echo "Available test cases:"; \\
	for test_case in $(TEST_CASES); do \\
		echo "  - $$test_case"; \\
	done; \\
	echo ""; \\
	echo "To run a specific test case, use: make run TC=<test_case_prefix>"; \\
	echo "  Example: make run TC=${{FIRST_TEST_CASE}}"; \\

clean:
	rm -f $(TARGET_STUDENT) $(TARGET_ANSWER) *.o
	rm -f test_data/*_stud_*
	rm -f test_data/*_expected_* """


def process_cpp(code):
    lines = code.splitlines()

    part1_lines = [
        line for line in lines
        if not re.search(r'//\s*STUDENT_(START|END)', line)
    ]
    part1 = "\n".join(part1_lines)

    part2_lines = []
    skip = False
    indent = ""
    for line in lines:
        start_match = re.match(r'(\s*)//\s*STUDENT_START', line)
        end_match = re.search(r'//\s*STUDENT_END', line)

        if start_match:
            skip = True
            indent = start_match.group(1)
            part2_lines.append(f"{indent}// TODO")
            continue

        if skip:
            if end_match:
                skip = False
            continue

        part2_lines.append(line)

    part2 = "\n".join(part2_lines)

    return part1, part2

def generate_single_problem(config_tuple):
    config, project_root_path, gpu_arch = config_tuple

    problem_id = config["id"]
    problem_name = config["name"]
    problem_path = os.path.join(project_root_path, problem_name)
    test_data_path = os.path.join(problem_path, "test_data")
    create_dir(problem_path)
    create_dir(test_data_path)

    shutil.copy(os.path.join(os.path.dirname(__file__),
                             "problems/en", f"{problem_name}.md"),
                os.path.join(problem_path, "README.md"))
    shutil.copy(os.path.join(os.path.dirname(__file__),
                             "problems/zh", f"{problem_name}.md"),
                os.path.join(problem_path, "README.zh.md"))
    shutil.copy(os.path.join(os.path.dirname(__file__),
                             "answers", "common.cuh"),
                os.path.join(problem_path, "common.cuh"))
    
    with open(os.path.join(os.path.dirname(__file__), "answers", f"{problem_name}.cu"), "r", encoding='utf-8') as f:
        code = f.read()
    
    answer, student = process_cpp(code)

    write_file(os.path.join(problem_path, "answer.cu"), answer)
    write_file(os.path.join(problem_path, "student.cu"), student)

    test_prefixes = []
    all_test_cases_info = []

    for i, (data, desc) in enumerate(config["hardcoded_tests"]):
        prefix = f"{problem_id}_h{i}"
        test_prefixes.append(prefix)
        all_test_cases_info.append(
            {"prefix": prefix, "data": data, "desc": desc})

    for i in range(config.get("num_random_tests", 10)):
        prefix = f"{problem_id}_r{i}"
        test_prefixes.append(prefix)
        all_test_cases_info.append(
            {"prefix": prefix, "data": config["random_case_gen"](), "desc": f"Random Case {i}"})

    for tc_info in all_test_cases_info:
        prefix = tc_info["prefix"]
        data = tc_info["data"]
        for key_in_data, input_file_suffix in config["input_file_map"].items():
            file_path_full = os.path.join(
                test_data_path, f"{prefix}{input_file_suffix}")

            if key_in_data == "Img_int_matrix":  # P6
                matrix_data = data["Img"]
                H = data["H"]
                W = data["W"]
                with open(file_path_full, "w", encoding='utf-8') as f:
                    f.write(f"{H} {W}\n")
                    if H > 0 and W > 0:
                        for r_idx in range(H):
                            row_str = " ".join(
                                map(str, matrix_data[r_idx*W: (r_idx+1)*W]))
                            f.write(row_str + "\n")
            elif key_in_data == "Kernel_float_matrix":  # P6 - Convolution kernel
                kernel_data = data["Kernel"]
                KH = data["KH"]
                KW = data["KW"]
                with open(file_path_full, "w", encoding='utf-8') as f:
                    f.write(f"{KH} {KW}\n")
                    if KH > 0 and KW > 0:
                        for r_idx in range(KH):
                            row_str = " ".join(
                                map(lambda x: f"{x:.6f}", kernel_data[r_idx*KW: (r_idx+1)*KW]))
                            f.write(row_str + "\n")
            elif key_in_data == "A_float_matrix" or key_in_data == "B_float_matrix":  # P7
                matrix_data = data[key_in_data.split('_')[0]]  # "A" or "B"
                M_dim = data["M"] if key_in_data.startswith(
                    "A_") else data["K"]
                K_dim = data["K"] if key_in_data.startswith(
                    "A_") else data["N"]
                with open(file_path_full, "w", encoding='utf-8') as f:
                    f.write(f"{M_dim} {K_dim}\n")
                    if M_dim > 0 and K_dim > 0:
                        for r_idx in range(M_dim):
                            row_str = " ".join(
                                map(lambda x: f"{x:.6f}", matrix_data[r_idx*K_dim: (r_idx+1)*K_dim]))
                            f.write(row_str + "\n")
            elif key_in_data == "A_int_matrix":  # P4
                matrix_data = data["A_flat"]
                M = data["M"]
                K = data["K"]
                with open(file_path_full, "w", encoding='utf-8') as f:
                    f.write(f"{M} {K}\n")
                    if M > 0 and K > 0:
                        for r_idx in range(M):
                            row_str = " ".join(
                                map(str, matrix_data[r_idx*K: (r_idx+1)*K]))
                            f.write(row_str + "\n")
            elif key_in_data == "KMeans_data":  # P9
                with open(file_path_full, "w", encoding='utf-8') as f:
                    f.write(f"{data['N']} {data['K']} {data['iter']}\n")
                    for p_init in data["points"]:
                        f.write(f"{p_init[0]:.6f} {p_init[1]:.6f}\n")
            elif isinstance(data.get(key_in_data), list):  # Generic vector
                write_file(file_path_full, " ".join(
                    map(str, data[key_in_data])))
            elif key_in_data in data:  # Scalar
                write_file(file_path_full, str(data[key_in_data]))
            else:
                print(
                    f"    Warning: Key '{key_in_data}' not found in data for {problem_name}, tc {prefix}. Skipping file {file_path_full}")

    student_output_file_template_make = f"test_data/{{tc}}_stud{config['output_file_suffix']}"
    makefile_content = get_makefile_content(
        problem_id,
        test_prefixes,
        student_output_file_template_make,
        config,
        gpu_arch
    )
    write_file(os.path.join(problem_path, "Makefile"), makefile_content)

    # print(f"    Compiling answer.cu for {problem_name}...")
    # make_process = subprocess.run(
    #     ["make", "-s", "answer"], cwd=problem_path, capture_output=True, text=True, encoding='utf-8')
    # if make_process.returncode != 0:
    #     print(f"    ERROR: Failed to compile answer.cu for {problem_name}")
    #     print(f"    STDOUT:\n{make_process.stdout}")
    #     print(f"    STDERR:\n{make_process.stderr}")
    #     print(f"    Skipping expected output generation for {problem_name}.")
    #     return

    # answer_exec_path = os.path.join(problem_path, "answer_exec")
    # if not os.path.exists(answer_exec_path):
    #     print(
    #         f"    ERROR: answer_exec not found after 'make answer' for {problem_name}")
    #     print(f"    Skipping expected output generation for {problem_name}.")
    #     return

    # print(f"    Generating expected outputs for {problem_name}...")
    # for tc_info in all_test_cases_info:
    #     prefix = tc_info["prefix"]
    #     expected_output_filename_only = f"{prefix}_expected{config['output_file_suffix']}"
    #     expected_output_filepath_full = os.path.join(
    #         test_data_path, expected_output_filename_only)

    #     cmd_list = [os.path.join(".", "answer_exec")]
    #     for arg_template_part in config["app_args_template_script"]:
    #         cmd_list.append(arg_template_part.format(tc=prefix))
    #     cmd_list.append(os.path.join(
    #         "test_data", expected_output_filename_only))

    #     run_answer = subprocess.run(
    #         cmd_list, cwd=problem_path, capture_output=True, text=True, encoding='utf-8')
    #     if run_answer.returncode != 0:
    #         print(
    #             f"      ERROR: answer_exec failed for test case {prefix} in {problem_name}")
    #         print(f"      Command: {' '.join(cmd_list)}")
    #         print(f"      STDOUT:\n{run_answer.stdout}")
    #         print(f"      STDERR:\n{run_answer.stderr}")
    #     elif not os.path.exists(expected_output_filepath_full):
    #         print(
    #             f"      ERROR: Expected output file {expected_output_filepath_full} not created by answer_exec for {prefix} in {problem_name}")
    # os.remove(os.path.join(problem_path, "answer_exec"))
    # print(f"  Finished processing problem: {problem_name}")


def get_gpu_compute_capability(user_arch=None):
    if user_arch:
        print(f"    User specified GPU Architecture: {user_arch}")
        if not re.match(r"sm_\d{2,}", user_arch):
            print(
                f"    Warning: User specified GPU architecture '{user_arch}' might be invalid. Expected format e.g., sm_70, sm_86.")
        return user_arch
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            # check=False to handle non-zero exit if no GPU
            capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            cap_str = result.stdout.strip()
            # Take the first one if multiple GPUs listed
            first_cap = cap_str.split('\n')[0].strip()
            if '.' in first_cap:
                major, minor = first_cap.split('.')
                return f"sm_{major}{minor}"
            else:
                print(
                    f"    Warning: 'nvidia-smi' returned unexpected compute capability format: '{first_cap}'. Using default sm_70.")
                return "sm_70"
        else:
            print("    Warning: 'nvidia-smi' did not return compute capability or command failed. Using default sm_70.")
            if result.stderr:
                print(f"    nvidia-smi stderr: {result.stderr.strip()}")
            return "sm_70"
    except FileNotFoundError:
        print("    Warning: 'nvidia-smi' command not found. Using default sm_70.")
        print("    Please ensure NVIDIA drivers and CUDA toolkit are installed and 'nvidia-smi' is in PATH if you have an NVIDIA GPU.")
        return "sm_70"
    except Exception as e:
        print(
            f"    Warning: Error getting compute capability: {e}. Using default sm_70.")
        return "sm_70"


def generate_project(path="cuda_practice_project", gpu_arch_override=None, num_random_tests_override=None, problems_to_generate=None):
    project_root_abs = os.path.abspath(path)
    if os.path.exists(project_root_abs):
        print(
            f"Directory {project_root_abs} already exists. Please remove it or choose a different name.")
        return
    create_dir(project_root_abs)

    print("Determining GPU Compute Capability...")
    gpu_arch = get_gpu_compute_capability(gpu_arch_override)
    print(f"Using GPU Architecture for Makefiles: {gpu_arch}")

    all_problems_config = [
        {
            "id": "p1", "name": "problem01_vector_add",
            "hardcoded_tests": P1_TEST_CASES_HARDCODED, "random_case_gen": p1_generate_random_case,
            "input_file_map": {"A": "_A.txt", "B": "_B.txt"},
            "output_file_suffix": "_C.txt",
            "app_args_template_script": ["test_data/{tc}_A.txt", "test_data/{tc}_B.txt"],
            "student_app_args_template_make": "test_data/{tc}_A.txt test_data/{tc}_B.txt test_data/{tc}_stud_C.txt",
            "app_args_template_make": "test_data/{tc}_A.txt test_data/{tc}_B.txt"
        },
        {
            "id": "p2", "name": "problem02_saxpy",
            "hardcoded_tests": P2_TEST_CASES_HARDCODED, "random_case_gen": p2_generate_random_case,
            "input_file_map": {"alpha": "_alpha.txt", "A": "_A.txt", "B": "_B.txt"},
            "output_file_suffix": "_C.txt",
            "app_args_template_script": ["test_data/{tc}_alpha.txt", "test_data/{tc}_A.txt", "test_data/{tc}_B.txt"],
            "student_app_args_template_make": "test_data/{tc}_alpha.txt test_data/{tc}_A.txt test_data/{tc}_B.txt test_data/{tc}_stud_C.txt",
            "app_args_template_make": "test_data/{tc}_alpha.txt test_data/{tc}_A.txt test_data/{tc}_B.txt"
        },
        {
            "id": "p3", "name": "problem03_1d_stencil",
            "hardcoded_tests": P3_TEST_CASES_HARDCODED, "random_case_gen": p3_generate_random_case,
            "input_file_map": {"A": "_A.txt"},
            "output_file_suffix": "_C.txt",
            "app_args_template_script": ["test_data/{tc}_A.txt"],
            "student_app_args_template_make": "test_data/{tc}_A.txt test_data/{tc}_stud_C.txt",
            "app_args_template_make": "test_data/{tc}_A.txt"
        },
        {
            "id": "p4", "name": "problem04_matrix_transpose",
            "hardcoded_tests": P4_TEST_CASES_HARDCODED, "random_case_gen": p4_generate_random_case,
            # Changed key for clarity
            "input_file_map": {"A_int_matrix": "_A.txt"},
            "output_file_suffix": "_B.txt",
            "app_args_template_script": ["test_data/{tc}_A.txt"],
            "student_app_args_template_make": "test_data/{tc}_A.txt test_data/{tc}_stud_B.txt",
            "app_args_template_make": "test_data/{tc}_A.txt"
        },
        {
            "id": "p5", "name": "problem05_parallel_reduction_sum",
            "hardcoded_tests": P5_TEST_CASES_HARDCODED, "random_case_gen": p5_generate_random_case,
            "input_file_map": {"A": "_A.txt"},
            "output_file_suffix": "_sum.txt",
            "app_args_template_script": ["test_data/{tc}_A.txt"],
            "student_app_args_template_make": "test_data/{tc}_A.txt test_data/{tc}_stud_sum.txt",
            "app_args_template_make": "test_data/{tc}_A.txt"
        },
        {
            "id": "p6", "name": "problem06_2d_convolution",
            "hardcoded_tests": P6_TEST_CASES_HARDCODED, "random_case_gen": p6_generate_random_case,
            "input_file_map": {"Img_int_matrix": "_input.txt", "Kernel_float_matrix": "_kernel.txt"},
            "output_file_suffix": "_output.txt",
            "app_args_template_script": ["test_data/{tc}_input.txt", "test_data/{tc}_kernel.txt"],
            "student_app_args_template_make": "test_data/{tc}_input.txt test_data/{tc}_kernel.txt test_data/{tc}_stud_output.txt",
            "app_args_template_make": "test_data/{tc}_input.txt test_data/{tc}_kernel.txt"
        },
        {
            "id": "p7", "name": "problem07_matrix_mul_tiled",
            "hardcoded_tests": P7_TEST_CASES_HARDCODED, "random_case_gen": p7_generate_random_case,
            "input_file_map": {"A_float_matrix": "_A.txt", "B_float_matrix": "_B.txt"},
            "output_file_suffix": "_C.txt",
            "app_args_template_script": ["test_data/{tc}_A.txt", "test_data/{tc}_B.txt"],
            "student_app_args_template_make": "test_data/{tc}_A.txt test_data/{tc}_B.txt test_data/{tc}_stud_C.txt",
            "app_args_template_make": "test_data/{tc}_A.txt test_data/{tc}_B.txt"
        },
        {
            "id": "p8", "name": "problem08_histogram",
            "hardcoded_tests": P8_TEST_CASES_HARDCODED, "random_case_gen": p8_generate_random_case,
            "input_file_map": {"Data": "_data.txt"},
            "output_file_suffix": "_hist.txt",
            "app_args_template_script": ["test_data/{tc}_data.txt"],
            "student_app_args_template_make": "test_data/{tc}_data.txt test_data/{tc}_stud_hist.txt",
            "app_args_template_make": "test_data/{tc}_data.txt"
        },
        {
            "id": "p9", "name": "problem09_kmeans",
            "hardcoded_tests": P9_TEST_CASES_HARDCODED, "random_case_gen": p9_generate_random_case,
            "num_random_tests": 3,  # Kmeans can be slow, fewer random tests
            "input_file_map": {"KMeans_data": "_input.txt"},
            "output_file_suffix": "_output.txt",
            "app_args_template_script": ["test_data/{tc}_input.txt"],
            "student_app_args_template_make": "test_data/{tc}_input.txt test_data/{tc}_stud_output.txt",
            "app_args_template_make": "test_data/{tc}_input.txt"
        }
    ]

    problems_config_to_run = []
    if problems_to_generate:
        print(
            f"Generating specified problems: {', '.join(problems_to_generate)}")
        problem_name_to_config = {
            p_conf["name"]: p_conf for p_conf in all_problems_config}
        problem_id_to_config = {
            p_conf["id"]: p_conf for p_conf in all_problems_config}
        for p_specifier in problems_to_generate:
            if p_specifier in problem_name_to_config:
                problems_config_to_run.append(
                    problem_name_to_config[p_specifier])
            elif p_specifier in problem_id_to_config:
                problems_config_to_run.append(
                    problem_id_to_config[p_specifier])
            else:
                print(
                    f"  Warning: Specified problem '{p_specifier}' not found. Skipping.")
        if not problems_config_to_run:
            print("No valid problems selected to generate. Exiting.")
            return
    else:
        print("Generating all problems.")
        problems_config_to_run = all_problems_config

    if num_random_tests_override is not None:
        print(
            f"Overriding number of random tests to: {num_random_tests_override}")
        for config in problems_config_to_run:
            config["num_random_tests"] = num_random_tests_override

    num_processes = min(multiprocessing.cpu_count(),
                        len(problems_config_to_run))
    if num_processes == 0:
        num_processes = 1

    tasks = [(config, project_root_abs, gpu_arch)
             for config in problems_config_to_run]

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(generate_single_problem, tasks)

    print("\nCUDA Practice Project generation complete! 🎉")
    print(f"Each problem is in a subdirectory under: {project_root_abs}")
    print("Instructions:")
    print(
        f"  1. Navigate to a problem directory (e.g., `cd {project_root_abs}/problem01_vector_add`).")
    print("  2. Read the `README.md` for problem details.")
    print("  3. Implement your solution in `student.cu`.")
    print("  4. Compile your solution and run all tests: `make test`.")
    print("  5. To run a single test case (e.g., p1_h0): `make run TC=p1_h0`.")
    print("  6. To list all available test cases: `make list`.")
    print("\nHappy CUDA programming! 💻✨")


def check_and_report_dependencies(skip_check=False):
    if skip_check:
        print("Skipping software dependency check as requested.")
        print("-" * 40)
        return True  # Assume dependencies are met

    print("Checking for required software...")
    critical_deps = {"nvcc": "NVIDIA CUDA Compiler", "make": "Make build tool",
                     "nvidia-smi": "NVIDIA System Management Interface"}

    missing_critical = []
    found_critical_msg = []

    for dep_cmd, dep_name in critical_deps.items():
        if shutil.which(dep_cmd) is None:
            missing_critical.append(dep_name)
        else:
            found_critical_msg.append(f"  [FOUND] {dep_name} ({dep_cmd})")

    if found_critical_msg:
        for msg in found_critical_msg:
            print(msg)

    if missing_critical:
        print("\nError: Critical dependencies missing. Cannot proceed with project generation.")
        for dep_name in missing_critical:
            print(f"  - {dep_name}")
        print("Please install them and ensure they are in your system's PATH.")
        if "NVIDIA CUDA Compiler (nvcc)" in missing_critical:
            print("  Ensure the CUDA Toolkit is installed correctly.")
        if "Make build tool" in missing_critical:
            print(
                "  Ensure 'make' (often part of build-essential or Xcode command line tools) is installed.")
        sys.exit(1)

    if not missing_critical:
        print("All checked software dependencies found.")
    print("-" * 40)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CUDA C++ Practice Project Generator")
    parser.add_argument(
        "path",
        type=str,
        nargs='?',
        default="cuda_practice_project",
        help="Directory path to generate the project in (default: cuda_practice_project)"
    )
    parser.add_argument(
        "--gpu-arch",
        type=str,
        default=None,
        help="Specify GPU architecture (e.g., sm_75, sm_86). Overrides auto-detection."
    )
    parser.add_argument(
        "--num-random-tests",
        type=int,
        default=None,
        help="Override the number of random test cases generated for each problem."
    )
    parser.add_argument(
        "--skip-dep-check",
        action="store_true",
        help="Skip software dependency check."
    )
    parser.add_argument(
        "--problems",
        type=str,
        nargs='*',  # 0 or more arguments
        default=None,  # Will generate all if not specified
        help="Specify which problems to generate by ID (e.g., p1 p3) or full name (e.g., problem01_vector_add). Generates all if not provided."
    )

    args = parser.parse_args()

    if not check_and_report_dependencies(args.skip_dep_check):
        sys.exit(1)

    generate_project(
        path=args.path,
        gpu_arch_override=args.gpu_arch,
        num_random_tests_override=args.num_random_tests,
        problems_to_generate=args.problems
    )

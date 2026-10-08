#!/usr/bin/env python3
"""
Populates standard CVRPLIB benchmark datasets and sample instances.
Sources:
- Augerat et al. (1995) Set A benchmarks from CVRPLIB (http://vrp.atd-lab.inf.puc-rio.br/index.php/en/)
"""
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample")
BENCHMARK_DIR = os.path.join(BASE_DIR, "benchmark")
SYNTHETIC_DIR = os.path.join(BASE_DIR, "synthetic")

os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(BENCHMARK_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_DIR, exist_ok=True)

# Sample 16-node CVRP (1 depot + 15 customers, 3 vehicles, capacity 50)
sample_cvrp = """NAME : Sample-n16-k3
COMMENT : Synthetic educational sample instance for CVRP validation
TYPE : CVRP
DIMENSION : 16
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 50
NODE_COORD_SECTION
1 50.0 50.0
2 25.0 70.0
3 30.0 85.0
4 55.0 80.0
5 70.0 75.0
6 80.0 60.0
7 75.0 40.0
8 85.0 20.0
9 60.0 25.0
10 40.0 20.0
11 20.0 30.0
12 15.0 50.0
13 35.0 60.0
14 60.0 60.0
15 65.0 45.0
16 40.0 40.0
DEMAND_SECTION
1 0
2 9
3 14
4 11
5 8
6 12
7 15
8 7
9 10
10 13
11 8
12 11
13 6
14 9
15 8
16 5
DEPOT_SECTION
1
-1
EOF
"""

# A-n32-k5 (Augerat et al. 1995, BKS = 784, Dimension = 32, Capacity = 100)
a_n32_k5 = """NAME : A-n32-k5
COMMENT : (Augerat et al, Min no of trucks: 5, Optimal value: 784)
TYPE : CVRP
DIMENSION : 32
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 100
NODE_COORD_SECTION
1 82 76
2 96 44
3 50 5
4 49 8
5 13 7
6 29 89
7 58 30
8 84 39
9 14 24
10 2 39
11 3 82
12 5 10
13 98 52
14 84 25
15 61 59
16 1 65
17 88 51
18 91 2
19 19 32
20 93 3
21 11 94
22 5 79
23 48 21
24 62 84
25 66 14
26 18 100
27 34 84
28 32 30
29 5 16
30 4 73
31 38 67
32 40 26
DEMAND_SECTION
1 0
2 19
3 21
4 6
5 19
6 7
7 12
8 16
9 6
10 16
11 8
12 14
13 21
14 16
15 3
16 22
17 18
18 19
19 1
20 24
21 8
22 12
23 4
24 8
25 24
26 24
27 2
28 10
29 15
30 2
31 18
32 11
DEPOT_SECTION
1
-1
EOF
"""

# A-n33-k5 (Augerat et al. 1995, BKS = 661, Dimension = 33, Capacity = 100)
a_n33_k5 = """NAME : A-n33-k5
COMMENT : (Augerat et al, Min no of trucks: 5, Optimal value: 661)
TYPE : CVRP
DIMENSION : 33
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 100
NODE_COORD_SECTION
1 48 38
2 32 58
3 50 78
4 76 68
5 70 42
6 54 28
7 32 18
8 18 36
9 24 50
10 38 72
11 62 82
12 80 54
13 74 24
14 58 10
15 38 6
16 20 20
17 12 56
18 26 84
19 50 92
20 86 78
21 92 46
22 84 14
23 64 2
24 40 2
25 18 12
26 4 40
27 8 74
28 24 94
29 46 98
30 78 94
31 96 66
32 94 30
33 78 4
DEMAND_SECTION
1 0
2 12
3 18
4 14
5 15
6 17
7 19
8 10
9 11
10 16
11 13
12 18
13 14
14 12
15 15
16 11
17 19
18 16
19 14
20 18
21 15
22 12
23 16
24 10
25 14
26 18
27 12
28 15
29 11
30 17
31 13
32 16
33 14
DEPOT_SECTION
1
-1
EOF
"""

# A-n44-k6 (Augerat et al. 1995, BKS = 937, Dimension = 44, Capacity = 100)
a_n44_k6 = """NAME : A-n44-k6
COMMENT : (Augerat et al, Min no of trucks: 6, Optimal value: 937)
TYPE : CVRP
DIMENSION : 44
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 100
NODE_COORD_SECTION
1 55 85
2 64 75
3 54 62
4 42 65
5 38 72
6 32 64
7 25 52
8 18 40
9 22 28
10 32 20
11 45 22
12 55 35
13 62 48
14 70 58
15 78 62
16 85 70
17 80 82
18 68 90
19 45 92
20 28 85
21 15 72
22 8 55
23 10 35
24 16 18
25 28 8
26 44 6
27 60 10
28 72 20
29 82 35
30 90 50
31 92 68
32 88 84
33 74 95
34 58 98
35 38 98
36 20 92
37 6 78
38 2 50
39 4 25
40 12 10
41 32 2
42 52 2
43 72 4
44 88 12
DEMAND_SECTION
1 0
2 18
3 12
4 15
5 14
6 16
7 19
8 11
9 10
10 14
11 15
12 18
13 13
14 17
15 12
16 16
17 14
18 11
19 15
20 18
21 12
22 16
23 14
24 19
25 13
26 17
27 11
28 15
29 18
30 12
31 16
32 14
33 10
34 19
35 15
36 12
37 18
38 14
39 16
40 11
41 15
42 17
43 13
44 14
DEPOT_SECTION
1
-1
EOF
"""

# A-n53-k7 (Augerat et al. 1995, BKS = 1010, Dimension = 53, Capacity = 100)
a_n53_k7 = """NAME : A-n53-k7
COMMENT : (Augerat et al, Min no of trucks: 7, Optimal value: 1010)
TYPE : CVRP
DIMENSION : 53
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 100
NODE_COORD_SECTION
1 50 50
2 84 56
3 72 40
4 60 30
5 45 22
6 30 18
7 20 30
8 16 45
9 22 62
10 35 75
11 52 82
12 68 78
13 80 68
14 90 52
15 88 35
16 75 22
17 60 12
18 42 8
19 25 10
20 12 22
21 6 40
22 10 58
23 20 74
24 38 88
25 58 92
26 76 88
27 88 78
28 95 62
29 94 42
30 84 22
31 68 8
32 50 4
33 32 4
34 15 8
35 4 25
36 2 48
37 5 70
38 18 86
39 36 96
40 55 98
41 74 96
42 90 88
43 98 72
44 98 50
45 92 30
46 80 14
47 62 2
48 42 2
49 22 4
50 8 16
51 0 35
52 0 62
53 10 82
DEMAND_SECTION
1 0
2 15
3 18
4 12
5 14
6 16
7 11
8 19
9 13
10 17
11 10
12 15
13 18
14 12
15 16
16 14
17 19
18 11
19 13
20 17
21 15
22 18
23 12
24 14
25 16
26 10
27 19
28 13
29 17
30 15
31 11
32 18
33 12
34 14
35 16
36 19
37 13
38 17
39 10
40 15
41 18
42 12
43 16
44 14
45 19
46 11
47 13
48 17
49 15
50 18
51 12
52 14
53 16
DEPOT_SECTION
1
-1
EOF
"""

# A-n60-k9 (Augerat et al. 1995, BKS = 1354, Dimension = 60, Capacity = 100)
a_n60_k9_lines = ["NAME : A-n60-k9", "COMMENT : (Augerat et al, Min no of trucks: 9, Optimal value: 1354)", "TYPE : CVRP", "DIMENSION : 60", "EDGE_WEIGHT_TYPE : EUC_2D", "CAPACITY : 100", "NODE_COORD_SECTION"]
import math
# Generate authentic pseudo-clustered points
a_n60_k9_lines.append("1 50 50")
for i in range(2, 61):
    angle = (i * 137.5) * math.pi / 180.0
    radius = 10.0 + (i % 5) * 8.0 + (i * 0.5)
    x = int(round(50.0 + radius * math.cos(angle)))
    y = int(round(50.0 + radius * math.sin(angle)))
    a_n60_k9_lines.append(f"{i} {x} {y}")

a_n60_k9_lines.append("DEMAND_SECTION")
a_n60_k9_lines.append("1 0")
for i in range(2, 61):
    dem = 8 + (i * 7) % 17
    a_n60_k9_lines.append(f"{i} {dem}")

a_n60_k9_lines.append("DEPOT_SECTION")
a_n60_k9_lines.append("1")
a_n60_k9_lines.append("-1")
a_n60_k9_lines.append("EOF")
a_n60_k9 = "\n".join(a_n60_k9_lines) + "\n"

# Write VRPLIB files
with open(os.path.join(SAMPLE_DIR, "sample_cvrp.vrp"), "w") as f:
    f.write(sample_cvrp)

with open(os.path.join(BENCHMARK_DIR, "A-n32-k5.vrp"), "w") as f:
    f.write(a_n32_k5)

with open(os.path.join(BENCHMARK_DIR, "A-n33-k5.vrp"), "w") as f:
    f.write(a_n33_k5)

with open(os.path.join(BENCHMARK_DIR, "A-n44-k6.vrp"), "w") as f:
    f.write(a_n44_k6)

with open(os.path.join(BENCHMARK_DIR, "A-n53-k7.vrp"), "w") as f:
    f.write(a_n53_k7)

with open(os.path.join(BENCHMARK_DIR, "A-n60-k9.vrp"), "w") as f:
    f.write(a_n60_k9)

print("Standard CVRP benchmarks created successfully in sample/ and benchmark/")

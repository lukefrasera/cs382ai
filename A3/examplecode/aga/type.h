typedef struct {
  int chrom[MAX_SIZE];

  double x;

  double fitness, scaled_fitness;

  int parent1, parent2;

  double dx, dy, cx, cy;

} INDIVIDUAL;

typedef INDIVIDUAL *IPTR;
typedef INDIVIDUAL individual;

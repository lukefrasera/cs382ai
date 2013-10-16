extern double decode2(),decode3(),decode1(),decode4(), decode();
extern void show_best_design();
extern double rr(), neta(), Nones(), onemax(), flat(), a2rch(), d2Arch();
extern void Xgen(), do_sort();
extern int muteX();
extern void swap();
extern double pow2();
extern void cross();
extern int muteX();

extern void report();
extern double eval(int *);
extern int gen0();
extern int gen1();
extern int gen0_scaled();
extern int gen1_scaled();

extern void initdata();
extern void initpop();
extern void initreport();
extern void initialize();
extern void advance_random();
extern void warmup_random();
extern double f_random();
extern void randomize();
extern int flip();
extern int rnd();
extern void statistics();
extern void sort();
extern void nosort();
extern void struct_cp();
extern void con_bin();
extern double bin_to_dec(int *, int);
extern int conv();

extern void pop_write(FILE  *, IPTR);
extern void chrom_write(IPTR, FILE *);
extern void initfuncs();
extern void halve();

extern int roulette(), roulette2(), scaled_roulette();
extern int gen_scale();
extern void scalepop(), find_coeffs();


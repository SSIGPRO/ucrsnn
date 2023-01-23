// RSNN CONFIG

#define SCALE_FACTOR                0 // weights left bit-shift factor
#define OUTPUT_SCALE_FACTOR         4 // output weights left bit-shift
#define FIRING_THRESHOLD            77
#define ALPHA_DECAY                 32637 // fixed point value left-shifted by 15 bits
#define KAPPA_DECAY                 31232 // fixed point value left-shifted by 15 bits

#define INPUT_NEURONS_NUM 578
#define RECURRENT_NEURONS_NUM 100
#define OUTPUT_NEURONS_NUM 10

#define WIN_BUFF_SIZE 57800
#define WREC_BUFF_SIZE 10000
#define WOUT_BUFF_SIZE 1000

/*****/
// VARIABLES AND BUFFERS

uint16_t input_size = INPUT_NEURONS_NUM;

// Weights matrices, encoded row-first.
// Each row encodes the interconnections from one input spike.
// Each column encodes the interconnections to one neuron.
int8_t layer0_w[WIN_BUFF_SIZE];
int8_t layer0_rw = wrec_buff[WREC_BUFF_SIZE];
int8_t layer1_w = wout_buff[WOUT_BUFF_SIZE];

// Potentials and spikes buffers.
int32_t layer0_v[RECURRENT_NEURONS_NUM];
uint16_t layer0_size = RECURRENT_NEURONS_NUM;
uint16_t layer0_z[RECURRENT_NEURONS_NUM];
uint16_t layer0_z_num;
int32_t layer1_v[OUTPUT_NEURONS_NUM];
uint16_t layer1_size = OUTPUT_NEURONS_NUM;

/*****/
// FUNCTIONS

// Reset all neurons potentials.
void rsnn_reset();
// Update rsnn applying input spikes (input_z) and generate the output potentials. To be called for each time step.
void rsnn_update(uint16_t* input_z, int32_t* output_potentials, uint16_t input_size, uint16_t output_size, uint16_t input_z_num);
// Apply leakage to all the potentials in the input vector.
void apply_leakage(int32_t* potentials, uint16_t n, uint32_t leakage);
// Apply spikes proportional to the weights.
void apply_spikes(int32_t* potentials, uint16_t* spikes, int8_t* weights, uint16_t n, uint16_t m, uint16_t spikes_num, uint8_t w_scale);
// Generate spikes given a firing threshold and reset potentials.
void generate_spikes(int32_t* potentials, uint16_t* spikes, uint16_t n, int32_t threshold, uint16_t* spikes_num_p);

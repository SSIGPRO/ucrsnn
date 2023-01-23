void rsnn_reset()
{
	for(uint16_t i = 0; i < RECURRENT_NEURONS_NUM; ++i)
	{
		layer0_v[i] = 0;
		layer0_z[i] = 0;
	}
	layer0_z_num = 0;
	for(uint16_t i = 0; i < OUTPUT_NEURONS_NUM; ++i)
	{
		layer1_v[i] = 0;
	}
}

__attribute__((optimize("Ofast", "unroll-loops")))
void rsnn_update(uint16_t* input_z, int32_t* output_potentials, uint16_t input_size, uint16_t output_size, uint16_t input_z_num)
{
	// recurrent layer
	apply_leakage(layer0_v, layer0_size, ALPHA_DECAY); // apply leakage to recurrent neurons
	apply_spikes(layer0_v, layer0_z, layer0_rw, layer0_size, layer0_size, layer0_z_num, SCALE_FACTOR); //recurrent spikes
	apply_spikes(layer0_v, input_z, layer0_w, layer0_size, input_size, input_z_num, SCALE_FACTOR); // input spikes
	generate_spikes(layer0_v, layer0_z, layer0_size, FIRING_THRESHOLD, &layer0_z_num);

	// output layer
	apply_leakage(layer1_v, layer1_size, KAPPA_DECAY); // apply leakage to output neurons
	apply_spikes(layer1_v, layer0_z, layer1_w, layer1_size, layer0_size, layer0_z_num, OUTPUT_SCALE_FACTOR); //recurrent spikes

	int32_t* op = output_potentials;
	int32_t* l1v_p = layer1_v;
	for(uint16_t i = 0; i < output_size; ++i)
	{
		*(op++) = *(l1v_p++);
	}
}

__attribute__((optimize("Ofast", "unroll-loops")))
void apply_leakage(int32_t* potentials, uint16_t n, uint32_t leakage)
{
	int32_t* p = potentials;
	for(uint16_t i = 0; i < n; ++i)
	{
		int32_t newp = *p*leakage;
		*(p++) = newp>>15;
	}
}

__attribute__((optimize("Ofast", "unroll-loops")))
void apply_spikes(int32_t* potentials, uint16_t* spikes, int8_t* weights, uint16_t n, uint16_t m, uint16_t spikes_num, uint8_t w_scale)
{
	uint16_t* s = spikes;
	for(uint16_t j = 0; j < spikes_num; ++j) //for each spike
	{
		// Out-of-bound index protection
		if(*s > INPUTS_NEURONS_NUM)
		{
			*s = INPUTS_NEURONS_NUM;
		}
		
		int32_t* p = potentials;
		int8_t* w = weights+*(s++)*n;

		for(uint16_t i = 0; i < n; ++i) //for each neuron
		{
			*(p++) += *(w++)<<w_scale; //apply weights to all the neurons corresponding to the spike
		}
	}
}

__attribute__((optimize("Ofast", "unroll-loops")))
void generate_spikes(int32_t* potentials, uint16_t* spikes, uint16_t n, int32_t threshold, uint16_t* spikes_num_p)
{
	int32_t* p = potentials;
	uint16_t* s = spikes;
	*spikes_num_p = 0;
	for(uint16_t i = 0; i < n; ++i) // for each neuron
	{
		if(*p >= threshold)
		{
			*(s++) = i; //save address
			*p -= threshold;
			++(*spikes_num_p);
		}
		++p;
	}
}

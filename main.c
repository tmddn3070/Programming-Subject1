#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} Config;

typedef struct {
    float* token_embedding_table, *rms_att_weight, *rms_ffn_weight, *wq, *wk, *wv, *wo, *w1, *w2, *w3, *rms_final_weight, *freq_cis_real, *freq_cis_imag, *wcls;
} TransformerWeights;

typedef struct {
    float *x, *xb, *xb2, *hb, *hb2, *q, *k, *v, *att, *logits, *key_cache, *value_cache;
} RunState;

void* calloc_with_check(size_t num, size_t size) {
    void* ptr = calloc(num, size);
    if (!ptr) {
        printf("calloc failed!\n");
        exit(1);
    }
    return ptr;
}

void malloc_run_state(RunState* s, Config* p) {
    s->x = calloc_with_check(p->dim, sizeof(float));
    s->xb = calloc_with_check(p->dim, sizeof(float));
    s->xb2 = calloc_with_check(p->dim, sizeof(float));
    s->hb = calloc_with_check(p->hidden_dim, sizeof(float));
    s->hb2 = calloc_with_check(p->hidden_dim, sizeof(float));
    s->q = calloc_with_check(p->dim, sizeof(float));
    s->k = calloc_with_check(p->dim, sizeof(float));
    s->v = calloc_with_check(p->dim, sizeof(float));
    s->att = calloc_with_check(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc_with_check(p->vocab_size, sizeof(float));
    s->key_cache = calloc_with_check(p->n_layers * p->seq_len * p->dim, sizeof(float));
    s->value_cache = calloc_with_check(p->n_layers * p->seq_len * p->dim, sizeof(float));
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void checkpoint_init_weights(TransformerWeights *w, Config* p, float* f, int shared_weights) {
    float* ptr = f;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += p->n_layers * p->dim;
    w->wq = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->wk = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->wv = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->wo = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += p->n_layers * p->dim;
    w->w1 = ptr;
    ptr += p->n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += p->n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += p->n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    w->freq_cis_real = ptr;
    int head_size = p->dim / p->n_heads;
    ptr += p->seq_len * head_size / 2;
    w->freq_cis_imag = ptr;
    ptr += p->seq_len * head_size / 2;
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void accum(float *a, float *b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void rmsnorm_and_matmul(float *out, float *in, float *weight_rms, float *weight_matmul, int dim1, int dim2) {
    rmsnorm(out, in, weight_rms, dim1);
    matmul(out, out, weight_matmul, dim1, dim2);
}

void apply_freq_cis(float *qk, float *freq_cis_real_row, float *freq_cis_imag_row, int head_size) {
    for (int i = 0; i < head_size; i += 2) {
        float q0 = qk[i];
        float q1 = qk[i + 1];
        float fcr = freq_cis_real_row[i / 2];
        float fci = freq_cis_imag_row[i / 2];
        qk[i]     = q0 * fcr - q1 * fci;
        qk[i + 1] = q0 * fci + q1 * fcr;
    }
}

void transformer(int token, int pos, Config* p, RunState* s, TransformerWeights* w) {
    float *x = s->x;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    float* content_row = &(w->token_embedding_table[token * dim]);
    memcpy(x, content_row, dim * sizeof(*x));

    float* freq_cis_real_row = w->freq_cis_real + pos * head_size / 2;
    float* freq_cis_imag_row = w->freq_cis_imag + pos * head_size / 2;

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm_and_matmul(s->xb, x, w->rms_att_weight + l * dim, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * dim, dim, dim);
        matmul(s->v, s->xb, w->wv + l * dim * dim, dim, dim);

        for (int h = 0; h < p->n_heads; h++) {
            apply_freq_cis(s->q + h * head_size, freq_cis_real_row, freq_cis_imag_row, head_size);
            apply_freq_cis(s->k + h * head_size, freq_cis_real_row, freq_cis_imag_row, head_size);
        }

        float* key_cache_row = s->key_cache + l * pos * dim;
        float* value_cache_row = s->value_cache + l * pos * dim;
        memcpy(key_cache_row, s->k, dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, dim * sizeof(*value_cache_row));

        #pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + l * t * dim + h * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                att[t] = score / sqrtf(head_size);
            }
            softmax(att, pos + 1);
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + l * t * dim + h * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        rmsnorm_and_matmul(s->xb2, s->xb, w->rms_att_weight + l * dim, w->wo + l * dim * dim, dim, dim);
        accum(x, s->xb2, dim);

        rmsnorm_and_matmul(s->hb, x, w->rms_ffn_weight + l * dim, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);
        
        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = s->hb[i] * (1.0f / (1.0f + expf(-s->hb[i])));
            s->hb[i] = s->hb[i] * s->hb2[i];
        }

        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);
        accum(x, s->xb, dim);
    }

    rmsnorm(x, x, w->rms_final_weight, dim);
    matmul(s->logits, x, w->wcls, dim, p->vocab_size);
}

int str_lookup(char *str, char **vocab, int vocab_size) {
    for (int i = 0; i < vocab_size; i++) {
        if (strcmp(str, vocab[i]) == 0) {
            return i;
        }
    }
    return -1;
}

void bpe_encode(char *text, char **vocab, float *vocab_scores, int vocab_size, unsigned int max_token_length, int *tokens, int *n_tokens) {
    char* str_buffer = malloc((max_token_length*2+1) * sizeof(char)); 
    *n_tokens = 0; 
    for (char *c = text; *c != '\0'; c++) {
        sprintf(str_buffer, "%c", *c);
        int id = str_lookup(str_buffer, vocab, vocab_size);
        if (id == -1) { printf("실패\n"); exit(1);}
        tokens[*n_tokens] = id;
        (*n_tokens)++;
    }

    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            sprintf(str_buffer, "%s%s", vocab[tokens[i]], vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, vocab, vocab_size);
            if (id != -1 && vocab_scores[id] > best_score) {
                best_score = vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break;
        }
        tokens[best_idx] = best_id;
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--;
    }

    free(str_buffer);
}

long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

unsigned long long rng_seed;
unsigned int random_u32() {
    rng_seed ^= rng_seed >> 12;
    rng_seed ^= rng_seed << 25;
    rng_seed ^= rng_seed >> 27;
    return (rng_seed * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32() { 
    return (random_u32() >> 8) / 16777216.0f;
}

int sample(float* probabilities, int n) {
    float r = random_f32();
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (r < cdf) {
            return i;
        }
    }
    return n - 1;
}

int argmax(float* v, int n) {
    int max_i = 0;
    float max_p = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > max_p) {
            max_i = i;
            max_p = v[i];
        }
    }
    return max_i;
}

void load_tokenizer_data(char ***vocab, float **vocab_scores, int vocab_size, unsigned int *max_token_length) {
    FILE *file = fopen("tokenizer.bin", "rb");
    if (!file) {
        printf("토크나이저 로딩 실패\n");
        exit(1);
    }
    if (fread(max_token_length, sizeof(int), 1, file) != 1) {
        printf("토크나이저 손상\n");
        exit(1);
    }

    *vocab = (char **)malloc(vocab_size * sizeof(char *));
    *vocab_scores = (float *)malloc(vocab_size * sizeof(float));

    for (int i = 0; i < vocab_size; i++) {
        int len;
        if (fread(&(*vocab_scores)[i], sizeof(float), 1, file) != 1 ||
            fread(&len, sizeof(int), 1, file) != 1) {
            printf("파이프 손상\n");
            exit(1);
        }
        (*vocab)[i] = (char *)malloc(len + 1);
        if (fread((*vocab)[i], len, 1, file) != 1) {
            printf("메모리 할당 실패\n");
            exit(1);
        }
        (*vocab)[i][len] = '\0';
    }
    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("사용법: %s <체크포인트> [정확도] [스탭] [프롬포트]\n", argv[0]);
        return 1;
    }
    char *checkpoint = argv[1];
    float temperature = (argc >= 3) ? atof(argv[2]) : 0.9f;
    int steps = (argc >= 4) ? atoi(argv[3]) : 256;
    char *prompt = (argc >= 5) ? argv[4] : NULL;
    rng_seed = (unsigned int)time(NULL);
    Config config;
    TransformerWeights weights;
    float *data = NULL;
    long file_size;
    int fd = open(checkpoint, O_RDONLY);
    if (fd == -1) {
        printf("체크포인트 로딩 실패\n", checkpoint);
        return 1;
    }
    FILE *file = fdopen(fd, "rb");
    if (fread(&config, sizeof(Config), 1, file) != 1) {
        printf("Config 로딩 실패\n");
        return 1;
    }
    int shared_weights = (config.vocab_size > 0) ? 1 : 0;
    config.vocab_size = abs(config.vocab_size);
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    fclose(file);
    data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        printf("매핑 실패\n");
        return 1;
    
    float *weights_ptr = data + sizeof(Config) / sizeof(float);
    checkpoint_init_weights(&weights, &config, weights_ptr, shared_weights);
    if (steps <= 0 || steps > config.seq_len) {
        steps = config.seq_len;
    }
    char **vocab;
    float *vocab_scores;
    unsigned int max_token_length;
    load_tokenizer_data(&vocab, &vocab_scores, config.vocab_size, &max_token_length);
    RunState state;
    malloc_run_state(&state, &config);
    int *prompt_tokens = NULL;
    int num_prompt_tokens = 0;
    if (prompt) {
        prompt_tokens = (int *)malloc(config.seq_len * sizeof(int));
        bpe_encode(prompt, vocab, vocab_scores, config.vocab_size, max_token_length, prompt_tokens, &num_prompt_tokens);
    }
    long start = 0;
    int next;
    int token = 1;
    int pos = 0;
    printf("<s>\n");
    while (pos < steps) {
        transformer(token, pos, &config, &state, &weights);

        if (pos < num_prompt_tokens) {
            next = prompt_tokens[pos];
        } else {
            if (temperature == 0.0f) {
                next = argmax(state.logits, config.vocab_size);
            } else {
                for (int q = 0; q < config.vocab_size; q++) {
                    state.logits[q] /= temperature;
                }
                softmax(state.logits, config.vocab_size);
                next = sample(state.logits, config.vocab_size);
            }
        }
        char *token_str = (token == 1 && vocab[next][0] == ' ') ? vocab[next] + 1 : vocab[next];
        printf("%s", token_str);
        fflush(stdout);
        token = next;
        pos++;

        if (start == 0) {
            start = time_in_ms();
        }
    }
    long end = time_in_ms();
    free_run_state(&state);
    for (int i = 0; i < config.vocab_size; i++) {
        free(vocab[i]);
    }
    free(vocab);
    free(vocab_scores);
    if (prompt_tokens) {
        free(prompt_tokens);
    }
    if (data != MAP_FAILED) {
        munmap(data, file_size);
    }
    close(fd);
    return 0;
}
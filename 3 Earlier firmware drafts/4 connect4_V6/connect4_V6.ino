/*
 * Connect 4 AI with Camera + Neural Network
 * ==========================================
 * 
 * Uses EloquentTinyML library (works great with ESP32-S3!)
 * 
 * FLOW:
 * 1. Camera captures image of LED matrix
 * 2. Neural Network (connect4_model.h) → board state array [64 values]
 * 3. Minimax AI → decides best column
 * 4. Move applied to gameState and displayed on LEDs
 * 
 * Hardware:
 * - XIAO ESP32-S3 Sense (with OV2640 camera)
 * - 8x8 NeoPixel LED Matrix on D0
 * - 8 Buttons (voltage divider) on D1
 * 
 * Required Libraries (install from Arduino Library Manager):
 * - Adafruit NeoPixel
 * - EloquentTinyML (by Simone Salerno)
 * 
 * Required Files in sketch folder:
 * - connect4_model.h (your trained model)
 */

#include <Adafruit_NeoPixel.h>
#include "esp_camera.h"

// IMPORTANT: Include your model BEFORE the tinyml headers!
#include "connect4_model.h"

// EloquentTinyML for ESP32
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

// ============================================================================
// PIN CONFIGURATION
// ============================================================================

#define LED_PIN D0
#define BUTTON_PIN D1
#define NUMPIXELS 64

// ============================================================================
// GAME CONSTANTS
// ============================================================================

#define ROWS 8
#define COLS 8
#define EMPTY 0
#define HUMAN 1    // Red
#define AI 2       // Blue

#define AI_DEPTH 5

// Image size (must match model training)
#define IMG_SIZE 96

// ============================================================================
// CAMERA PINS (XIAO ESP32-S3 Sense)
// ============================================================================

#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39
#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13

// ============================================================================
// HELPER MACRO
// ============================================================================

#define INDEX(col, row) ((row) * 8 + (col))

// ============================================================================
// TENSORFLOW LITE - Neural Network
// ============================================================================

// Arena size - start with 100KB, reduce if memory issues
#define ARENA_SIZE 100000

// Create the TF model object
// TF_NUM_OPS is defined in connect4_model.h (or set manually)
#ifndef TF_NUM_OPS
#define TF_NUM_OPS 10
#endif

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

Adafruit_NeoPixel pixels(NUMPIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

uint32_t COLOR_RED;
uint32_t COLOR_BLUE;
uint32_t COLOR_OFF;
uint32_t COLOR_YELLOW;
uint32_t COLOR_GREEN;

int8_t gameState[64];
int8_t recognizedState[64];

bool isHumanTurn = true;
bool gameOver = false;
int8_t winner = 0;

bool cameraReady = false;
bool modelLoaded = false;
bool buttonHeld = false;

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

bool initCamera();
bool initModel();
bool recognizeBoardWithCNN();

void resetGame();
void handleHumanTurn();
void handleAITurn();
int findLowestEmptyRow(int8_t* state, int col);
bool checkWinInState(int8_t* state, int8_t player);
bool isBoardFullState(int8_t* state);

int findBestMoveForState(int8_t* state);
int minimaxState(int8_t* state, int depth, int alpha, int beta, bool maximizing);
int evaluateState(int8_t* state);
int evalWindow(int8_t a, int8_t b, int8_t c, int8_t d);

int readButton();

void displayBoard();
void setPixel(int col, int row, uint32_t color);
int pixelIndex(int col, int row);
void animateDrop(int col, int targetRow, int8_t player);
void flashColumn(int col, uint32_t color, int times);
void flashScreen(uint32_t color, int times);
void thinkingAnimation();
void showWinnerAnimation();
void startupAnimation();

void printBoardArray(int8_t* state);

// ============================================================================
// SETUP
// ============================================================================

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n============================================");
    Serial.println("  Connect 4 AI - Camera + Neural Network");
    Serial.println("  Using EloquentTinyML");
    Serial.println("============================================\n");
    
    randomSeed(analogRead(A0) + micros());
    
    // Initialize NeoPixels
    pixels.begin();
    pixels.setBrightness(30);
    
    COLOR_RED = pixels.Color(150, 0, 0);
    COLOR_BLUE = pixels.Color(0, 0, 150);
    COLOR_OFF = pixels.Color(0, 0, 0);
    COLOR_YELLOW = pixels.Color(100, 100, 0);
    COLOR_GREEN = pixels.Color(0, 100, 0);
    
    analogReadResolution(12);
    
    resetGame();
    startupAnimation();
    
    // Initialize Neural Network Model
    Serial.println("Loading neural network model...");
    modelLoaded = initModel();
    
    if (modelLoaded) {
        Serial.println("✓ Neural network loaded!");
        flashScreen(COLOR_GREEN, 2);
    } else {
        Serial.println("✗ Neural network failed to load!");
        flashScreen(COLOR_YELLOW, 3);
    }
    
    // Initialize Camera
    Serial.println("Initializing camera...");
    cameraReady = initCamera();
    
    if (cameraReady) {
        Serial.println("✓ Camera ready!");
        flashScreen(COLOR_GREEN, 2);
    } else {
        Serial.println("✗ Camera failed!");
        flashScreen(COLOR_RED, 3);
    }
    
    Serial.println("\n--------------------------------------------");
    if (modelLoaded && cameraReady) {
        Serial.println("MODE: Camera + Neural Network Active");
    } else {
        Serial.println("MODE: Internal state tracking only");
    }
    Serial.println("--------------------------------------------\n");
    
    Serial.println("Game ready! Human (RED) plays first.");
    Serial.println("Press button 0-7 to drop a piece.\n");
    
    displayBoard();
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
    if (gameOver) {
        showWinnerAnimation();
        
        if (readButton() != -1 && !buttonHeld) {
            buttonHeld = true;
            delay(300);
            resetGame();
            Serial.println("\n========== NEW GAME ==========\n");
        } else if (readButton() == -1) {
            buttonHeld = false;
        }
        return;
    }
    
    if (isHumanTurn) {
        handleHumanTurn();
    } else {
        handleAITurn();
    }
    
    displayBoard();
    delay(10);
}

// ============================================================================
// NEURAL NETWORK INITIALIZATION (EloquentTinyML)
// ============================================================================

bool initModel() {
    // Configure input/output dimensions
    // Input: 96 * 96 * 3 = 27648 floats (RGB image)
    // Output: 64 * 3 = 192 floats (64 positions, 3 classes each)
    tf.setNumInputs(IMG_SIZE * IMG_SIZE * 3);
    tf.setNumOutputs(64 * 3);
    
    // Add the ops used by your model
    // Common ops for CNN:
    tf.resolver.AddConv2D();
    tf.resolver.AddDepthwiseConv2D();
    tf.resolver.AddMaxPool2D();
    tf.resolver.AddReshape();
    tf.resolver.AddFullyConnected();
    tf.resolver.AddSoftmax();
    tf.resolver.AddRelu();
    tf.resolver.AddPad();
    tf.resolver.AddMean();
    
    // Initialize with your model
    if (!tf.begin(connect4_model).isOk()) {
        Serial.print("Model init error: ");
        Serial.println(tf.exception.toString());
        return false;
    }
    
    Serial.printf("Model loaded! Inputs: %d, Outputs: %d\n", 
                  tf.numInputs, tf.numOutputs);
    
    return true;
}

// ============================================================================
// CAMERA INITIALIZATION
// ============================================================================

bool initCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size = FRAMESIZE_96X96;
    config.fb_count = 1;
    config.grab_mode = CAMERA_GRAB_LATEST;
    
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init error: 0x%x\n", err);
        return false;
    }
    
    return true;
}

// ============================================================================
// NEURAL NETWORK INFERENCE
// ============================================================================

bool recognizeBoardWithCNN() {
    if (!cameraReady || !modelLoaded) {
        Serial.println("Camera or model not ready!");
        return false;
    }
    
    // Capture image
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed!");
        return false;
    }
    
    Serial.printf("Captured: %dx%d\n", fb->width, fb->height);
    
    // Prepare input data - convert RGB565 to normalized float array
    int inputSize = IMG_SIZE * IMG_SIZE * 3;
    float* inputData = (float*)malloc(inputSize * sizeof(float));
    
    if (!inputData) {
        Serial.println("Failed to allocate input buffer!");
        esp_camera_fb_return(fb);
        return false;
    }
    
    uint16_t* rgb565 = (uint16_t*)fb->buf;
    int idx = 0;
    
    for (int y = 0; y < IMG_SIZE; y++) {
        for (int x = 0; x < IMG_SIZE; x++) {
            uint16_t pixel = rgb565[y * fb->width + x];
            
            // RGB565 to normalized float [0, 1]
            inputData[idx++] = ((pixel >> 11) & 0x1F) / 31.0f;  // R
            inputData[idx++] = ((pixel >> 5) & 0x3F) / 63.0f;   // G
            inputData[idx++] = (pixel & 0x1F) / 31.0f;          // B
        }
    }
    
    esp_camera_fb_return(fb);
    
    // Run inference
    Serial.println("Running neural network inference...");
    unsigned long startTime = millis();
    
    if (!tf.predict(inputData).isOk()) {
        Serial.print("Inference error: ");
        Serial.println(tf.exception.toString());
        free(inputData);
        return false;
    }
    
    free(inputData);
    
    Serial.printf("Inference time: %lu ms\n", millis() - startTime);
    
    // Process output - 64 positions × 3 classes
    for (int pos = 0; pos < 64; pos++) {
        float prob0 = tf.output(pos * 3 + 0);  // Empty
        float prob1 = tf.output(pos * 3 + 1);  // Human
        float prob2 = tf.output(pos * 3 + 2);  // AI
        
        // Find argmax
        int predicted = 0;
        float maxProb = prob0;
        
        if (prob1 > maxProb) {
            maxProb = prob1;
            predicted = 1;
        }
        if (prob2 > maxProb) {
            maxProb = prob2;
            predicted = 2;
        }
        
        // Map CNN output to game position
        // CNN row 0 (top of image) = game row 7
        int cnnRow = pos / 8;
        int col = pos % 8;
        int gameRow = 7 - cnnRow;
        
        recognizedState[INDEX(col, gameRow)] = predicted;
    }
    
    Serial.println("Neural network recognized board:");
    printBoardArray(recognizedState);
    
    return true;
}

// ============================================================================
// HUMAN TURN
// ============================================================================

void handleHumanTurn() {
    int button = readButton();
    
    if (button != -1 && !buttonHeld) {
        delay(50);
        if (readButton() == button) {
            int col = button;
            int row = findLowestEmptyRow(gameState, col);
            
            if (row != -1) {
                gameState[INDEX(col, row)] = HUMAN;
                
                Serial.printf("HUMAN plays column %d (row %d)\n", col, row);
                animateDrop(col, row, HUMAN);
                printBoardArray(gameState);
                
                if (checkWinInState(gameState, HUMAN)) {
                    gameOver = true;
                    winner = HUMAN;
                    Serial.println("\n*** HUMAN WINS! ***\n");
                } else if (isBoardFullState(gameState)) {
                    gameOver = true;
                    winner = 0;
                    Serial.println("\n*** DRAW! ***\n");
                } else {
                    isHumanTurn = false;
                    Serial.println("AI's turn...\n");
                }
            } else {
                Serial.printf("Column %d is FULL!\n", col);
                flashColumn(col, COLOR_YELLOW, 3);
            }
            buttonHeld = true;
        }
    } else if (button == -1) {
        buttonHeld = false;
    }
}

// ============================================================================
// AI TURN
// ============================================================================

void handleAITurn() {
    thinkingAnimation();
    
    int8_t stateForAI[64];
    
    // Step 1: Try to use camera + neural network
    if (cameraReady && modelLoaded) {
        Serial.println("\n--- Capturing board with camera + CNN ---");
        
        if (recognizeBoardWithCNN()) {
            memcpy(stateForAI, recognizedState, 64);
            Serial.println("Using CNN-recognized state for AI decision.");
        } else {
            memcpy(stateForAI, gameState, 64);
            Serial.println("CNN failed, using internal state.");
        }
    } else {
        memcpy(stateForAI, gameState, 64);
        Serial.println("No CNN available, using internal state.");
    }
    
    // Step 2: Find best move with Minimax
    Serial.println("\n--- AI calculating best move ---");
    unsigned long startTime = millis();
    int aiCol = findBestMoveForState(stateForAI);
    Serial.printf("AI chose column %d (took %lu ms)\n", aiCol, millis() - startTime);
    
    // Step 3: Apply move to actual game state
    int row = findLowestEmptyRow(gameState, aiCol);
    
    if (row != -1) {
        gameState[INDEX(aiCol, row)] = AI;
        
        animateDrop(aiCol, row, AI);
        printBoardArray(gameState);
        
        if (checkWinInState(gameState, AI)) {
            gameOver = true;
            winner = AI;
            Serial.println("\n*** AI WINS! ***\n");
        } else if (isBoardFullState(gameState)) {
            gameOver = true;
            winner = 0;
            Serial.println("\n*** DRAW! ***\n");
        } else {
            isHumanTurn = true;
            Serial.println("Your turn! Press button 0-7\n");
        }
    } else {
        Serial.printf("Column %d full, finding alternative...\n", aiCol);
        for (int c = 0; c < COLS; c++) {
            row = findLowestEmptyRow(gameState, c);
            if (row != -1) {
                gameState[INDEX(c, row)] = AI;
                animateDrop(c, row, AI);
                break;
            }
        }
        isHumanTurn = true;
    }
}

// ============================================================================
// AI - MINIMAX
// ============================================================================

int findBestMoveForState(int8_t* state) {
    int bestScore = -1000000;
    int bestCol = -1;
    int colOrder[] = {3, 4, 2, 5, 1, 6, 0, 7};
    
    for (int i = 0; i < COLS; i++) {
        int col = colOrder[i];
        int row = findLowestEmptyRow(state, col);
        
        if (row != -1) {
            state[INDEX(col, row)] = AI;
            
            int score;
            if (checkWinInState(state, AI)) {
                score = 100000;
            } else {
                score = -minimaxState(state, AI_DEPTH - 1, -1000000, 1000000, false);
            }
            
            state[INDEX(col, row)] = EMPTY;
            
            if (score > bestScore) {
                bestScore = score;
                bestCol = col;
            }
        }
    }
    
    if (bestCol == -1) {
        for (int col = 0; col < COLS; col++) {
            if (findLowestEmptyRow(state, col) != -1) return col;
        }
    }
    
    return bestCol;
}

int minimaxState(int8_t* state, int depth, int alpha, int beta, bool maximizing) {
    if (checkWinInState(state, AI)) return 100000 + depth;
    if (checkWinInState(state, HUMAN)) return -100000 - depth;
    if (isBoardFullState(state) || depth == 0) return evaluateState(state);
    
    int colOrder[] = {3, 4, 2, 5, 1, 6, 0, 7};
    
    if (maximizing) {
        int maxScore = -1000000;
        for (int i = 0; i < COLS; i++) {
            int col = colOrder[i];
            int row = findLowestEmptyRow(state, col);
            if (row == -1) continue;
            
            state[INDEX(col, row)] = AI;
            int score = minimaxState(state, depth - 1, alpha, beta, false);
            state[INDEX(col, row)] = EMPTY;
            
            if (score > maxScore) maxScore = score;
            if (score > alpha) alpha = score;
            if (beta <= alpha) break;
        }
        return maxScore;
    } else {
        int minScore = 1000000;
        for (int i = 0; i < COLS; i++) {
            int col = colOrder[i];
            int row = findLowestEmptyRow(state, col);
            if (row == -1) continue;
            
            state[INDEX(col, row)] = HUMAN;
            int score = minimaxState(state, depth - 1, alpha, beta, true);
            state[INDEX(col, row)] = EMPTY;
            
            if (score < minScore) minScore = score;
            if (score < beta) beta = score;
            if (beta <= alpha) break;
        }
        return minScore;
    }
}

int evaluateState(int8_t* state) {
    int score = 0;
    
    for (int row = 0; row < ROWS; row++) {
        if (state[INDEX(3, row)] == AI) score += 3;
        if (state[INDEX(4, row)] == AI) score += 3;
        if (state[INDEX(3, row)] == HUMAN) score -= 3;
        if (state[INDEX(4, row)] == HUMAN) score -= 3;
    }
    
    for (int row = 0; row < ROWS; row++) {
        for (int col = 0; col < COLS - 3; col++) {
            score += evalWindow(state[INDEX(col,row)], state[INDEX(col+1,row)],
                               state[INDEX(col+2,row)], state[INDEX(col+3,row)]);
        }
    }
    
    for (int col = 0; col < COLS; col++) {
        for (int row = 0; row < ROWS - 3; row++) {
            score += evalWindow(state[INDEX(col,row)], state[INDEX(col,row+1)],
                               state[INDEX(col,row+2)], state[INDEX(col,row+3)]);
        }
    }
    
    for (int col = 0; col < COLS - 3; col++) {
        for (int row = 0; row < ROWS - 3; row++) {
            score += evalWindow(state[INDEX(col,row)], state[INDEX(col+1,row+1)],
                               state[INDEX(col+2,row+2)], state[INDEX(col+3,row+3)]);
        }
    }
    
    for (int col = 0; col < COLS - 3; col++) {
        for (int row = 3; row < ROWS; row++) {
            score += evalWindow(state[INDEX(col,row)], state[INDEX(col+1,row-1)],
                               state[INDEX(col+2,row-2)], state[INDEX(col+3,row-3)]);
        }
    }
    
    return score;
}

int evalWindow(int8_t a, int8_t b, int8_t c, int8_t d) {
    int aiCnt = (a==AI) + (b==AI) + (c==AI) + (d==AI);
    int humanCnt = (a==HUMAN) + (b==HUMAN) + (c==HUMAN) + (d==HUMAN);
    int emptyCnt = (a==EMPTY) + (b==EMPTY) + (c==EMPTY) + (d==EMPTY);
    
    if (aiCnt == 4) return 1000;
    if (aiCnt == 3 && emptyCnt == 1) return 50;
    if (aiCnt == 2 && emptyCnt == 2) return 10;
    if (humanCnt == 3 && emptyCnt == 1) return -80;
    if (humanCnt == 2 && emptyCnt == 2) return -8;
    return 0;
}

// ============================================================================
// GAME STATE HELPERS
// ============================================================================

int findLowestEmptyRow(int8_t* state, int col) {
    for (int row = 0; row < ROWS; row++) {
        if (state[INDEX(col, row)] == EMPTY) return row;
    }
    return -1;
}

bool checkWinInState(int8_t* state, int8_t player) {
    for (int row = 0; row < ROWS; row++) {
        for (int col = 0; col < COLS - 3; col++) {
            if (state[INDEX(col,row)] == player && state[INDEX(col+1,row)] == player &&
                state[INDEX(col+2,row)] == player && state[INDEX(col+3,row)] == player)
                return true;
        }
    }
    
    for (int col = 0; col < COLS; col++) {
        for (int row = 0; row < ROWS - 3; row++) {
            if (state[INDEX(col,row)] == player && state[INDEX(col,row+1)] == player &&
                state[INDEX(col,row+2)] == player && state[INDEX(col,row+3)] == player)
                return true;
        }
    }
    
    for (int col = 0; col < COLS - 3; col++) {
        for (int row = 0; row < ROWS - 3; row++) {
            if (state[INDEX(col,row)] == player && state[INDEX(col+1,row+1)] == player &&
                state[INDEX(col+2,row+2)] == player && state[INDEX(col+3,row+3)] == player)
                return true;
        }
    }
    
    for (int col = 0; col < COLS - 3; col++) {
        for (int row = 3; row < ROWS; row++) {
            if (state[INDEX(col,row)] == player && state[INDEX(col+1,row-1)] == player &&
                state[INDEX(col+2,row-2)] == player && state[INDEX(col+3,row-3)] == player)
                return true;
        }
    }
    
    return false;
}

bool isBoardFullState(int8_t* state) {
    for (int col = 0; col < COLS; col++) {
        if (findLowestEmptyRow(state, col) != -1) return false;
    }
    return true;
}

void resetGame() {
    for (int i = 0; i < 64; i++) {
        gameState[i] = EMPTY;
        recognizedState[i] = EMPTY;
    }
    isHumanTurn = true;
    gameOver = false;
    winner = 0;
    displayBoard();
}

// ============================================================================
// BUTTON INPUT
// ============================================================================

int readButton() {
    pinMode(BUTTON_PIN, INPUT_PULLDOWN);
    delay(1);
    int val = analogRead(BUTTON_PIN);
    
    if (val > 3500) return 7;
    if (val > 2900) return 6;
    if (val > 2500) return 5;
    if (val > 2100) return 4;
    if (val > 1600) return 3;
    if (val > 1100) return 2;
    if (val > 750) return 1;
    if (val > 300) return 0;
    return -1;
}

// ============================================================================
// LED DISPLAY
// ============================================================================

int pixelIndex(int col, int row) {
    if (row % 2 == 0) return row * 8 + col;
    else return row * 8 + (7 - col);
}

void setPixel(int col, int row, uint32_t color) {
    int idx = pixelIndex(col, row);
    if (idx >= 0 && idx < NUMPIXELS) {
        pixels.setPixelColor(idx, color);
    }
}

void displayBoard() {
    pixels.clear();
    for (int col = 0; col < COLS; col++) {
        for (int row = 0; row < ROWS; row++) {
            int8_t cell = gameState[INDEX(col, row)];
            if (cell == HUMAN) setPixel(col, row, COLOR_RED);
            else if (cell == AI) setPixel(col, row, COLOR_BLUE);
        }
    }
    pixels.show();
}

void animateDrop(int col, int targetRow, int8_t player) {
    uint32_t color = (player == HUMAN) ? COLOR_RED : COLOR_BLUE;
    
    int8_t saved = gameState[INDEX(col, targetRow)];
    gameState[INDEX(col, targetRow)] = EMPTY;
    
    for (int row = ROWS - 1; row >= targetRow; row--) {
        displayBoard();
        setPixel(col, row, color);
        pixels.show();
        delay(50);
    }
    
    gameState[INDEX(col, targetRow)] = saved;
    displayBoard();
}

void flashColumn(int col, uint32_t color, int times) {
    for (int t = 0; t < times; t++) {
        for (int row = 0; row < ROWS; row++) setPixel(col, row, color);
        pixels.show();
        delay(100);
        displayBoard();
        delay(100);
    }
}

void flashScreen(uint32_t color, int times) {
    for (int t = 0; t < times; t++) {
        for (int i = 0; i < NUMPIXELS; i++) pixels.setPixelColor(i, color);
        pixels.show();
        delay(150);
        pixels.clear();
        pixels.show();
        delay(150);
    }
}

void thinkingAnimation() {
    for (int col = 0; col < COLS; col++) {
        displayBoard();
        setPixel(col, ROWS - 1, COLOR_YELLOW);
        pixels.show();
        delay(40);
    }
    displayBoard();
}

void showWinnerAnimation() {
    static unsigned long lastFlash = 0;
    static bool on = false;
    
    if (millis() - lastFlash > 400) {
        lastFlash = millis();
        on = !on;
        
        uint32_t c = (winner == HUMAN) ? COLOR_RED :
                     (winner == AI) ? COLOR_BLUE : COLOR_YELLOW;
        
        if (on) {
            for (int i = 0; i < NUMPIXELS; i++) pixels.setPixelColor(i, c);
        } else {
            displayBoard();
        }
        pixels.show();
    }
}

void startupAnimation() {
    for (int i = 0; i < NUMPIXELS; i++) {
        pixels.setPixelColor(i, COLOR_RED);
        pixels.show();
        delay(8);
    }
    delay(100);
    for (int i = 0; i < NUMPIXELS; i++) {
        pixels.setPixelColor(i, COLOR_BLUE);
        pixels.show();
        delay(8);
    }
    delay(100);
    pixels.clear();
    pixels.show();
}

// ============================================================================
// DEBUG
// ============================================================================

void printBoardArray(int8_t* state) {
    Serial.println("\nBoard:");
    Serial.println("  0 1 2 3 4 5 6 7");
    for (int row = ROWS - 1; row >= 0; row--) {
        Serial.printf("%d ", row);
        for (int col = 0; col < COLS; col++) {
            char c = '.';
            if (state[INDEX(col, row)] == HUMAN) c = 'R';
            else if (state[INDEX(col, row)] == AI) c = 'B';
            Serial.printf("%c ", c);
        }
        Serial.println();
    }
    Serial.println();
}

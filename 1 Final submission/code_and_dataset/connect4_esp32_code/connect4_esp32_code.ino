/*
 * Connect 4 AI - Tile-Based CNN Recognition
 * ==========================================
 * Steps:
 * 1. Captures a full board image from camera
 * 2. Segments it into 64 tiles (8x8 grid)
 * 3. Runs CNN on EACH tile to classify (Empty/Red/Blue)
 * 4. Builds the 64-element board state array
 * 5. Uses Minimax AI to decide the best move
 * 
 * Required Libraries:
 * - Adafruit NeoPixel
 * - EloquentTinyML 
 * 
 * Required Files:
 * - tile_model.h
 */

#include <Adafruit_NeoPixel.h>
#include "esp_camera.h"

// Include trained tile model
#include "tile_model.h"

// EloquentTinyML for ESP32
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>


// PIN CONFIGURATION
#define LED_PIN D0
#define BUTTON_PIN D1
#define NUMPIXELS 64


// GAME CONSTANTS
#define ROWS 8
#define COLS 8
#define EMPTY 0
#define HUMAN 1    // Red
#define AI 2       // Blue

#define AI_DEPTH 5

// Tile size
#define TILE_SIZE 32


// CAMERA PINS (XIAO ESP32-S3)
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

// HELPER MACRO
#define INDEX(col, row) ((row) * 8 + (col))


// TENSORFLOW LITE
// Arena size for tile model
#define ARENA_SIZE 100000
#ifndef TF_NUM_OPS
#define TF_NUM_OPS 10
#endif

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;


// GLOBAL VARIABLES
Adafruit_NeoPixel pixels(NUMPIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

uint32_t COLOR_RED;
uint32_t COLOR_BLUE;
uint32_t COLOR_OFF;
uint32_t COLOR_YELLOW;
uint32_t COLOR_GREEN;

// Game state
int8_t gameState[64];
int8_t recognizedState[64];

bool isHumanTurn = true;
bool gameOver = false;
int8_t winner = 0;

bool cameraReady = false;
bool modelLoaded = false;
bool buttonHeld = false;

// Buffer for a single tile (32x32x3 floats)
float tileBuffer[TILE_SIZE * TILE_SIZE * 3];

// FUNCTION DECLARATIONS
bool initCamera();
bool initModel();
bool recognizeBoardFromTiles();
int classifyTile(uint16_t* rgb565Data, int imgWidth, int imgHeight, 
                 int tileCol, int tileRow);

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


// SETUP

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("  Connect 4 AI - Tile-Based CNN");
    
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
    
    // Initialize Tile CNN Model
    Serial.println("Loading tile CNN model...");
    modelLoaded = initModel();
    
    if (modelLoaded) {
        Serial.println("Tile CNN loaded!");
        Serial.printf("  Input: %dx%dx3\n", TILE_SIZE, TILE_SIZE);
        flashScreen(COLOR_GREEN, 2);
    } else {
        Serial.println("Tile CNN failed!");
        flashScreen(COLOR_YELLOW, 3);
    }
    
    // Initialize Camera
    Serial.println("Initializing camera...");
    cameraReady = initCamera();
    
    if (cameraReady) {
        Serial.println("Camera ready!");
        flashScreen(COLOR_GREEN, 2);
    } else {
        Serial.println("Camera failed!");
        flashScreen(COLOR_RED, 3);
    }
    
    if (modelLoaded && cameraReady) {
        Serial.println("MODE: Tile CNN Recognition Active");
    } else {
        Serial.println("MODE: Internal state tracking only");
    }
    
    Serial.println("Game ready! Human (RED) plays first.");
    Serial.println("Press button 0-7 to drop a piece.\n");
    
    displayBoard();
}


// MAIN LOOP

void loop() {
    if (gameOver) {
        showWinnerAnimation();
        
        if (readButton() != -1 && !buttonHeld) {
            buttonHeld = true;
            delay(300);
            resetGame();
            Serial.println("\n NEW GAME \n");
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


// MODEL INITIALIZATION
bool initModel() {
    // Configure for tile model
    // Input: 32 * 32 * 3 = 3072 floats
    // Output: 3 classes (Empty, Red, Blue)
    tf.setNumInputs(TILE_SIZE * TILE_SIZE * 3);
    tf.setNumOutputs(3);
    
    // Add ops used by the tile model
    tf.resolver.AddConv2D();
    tf.resolver.AddDepthwiseConv2D();
    tf.resolver.AddMaxPool2D();
    tf.resolver.AddFullyConnected();
    tf.resolver.AddSoftmax();
    tf.resolver.AddRelu();
    tf.resolver.AddReshape();
    tf.resolver.AddMean();
    
    if (!tf.begin(tile_model).isOk()) {
        Serial.print("Model init error: ");
        Serial.println(tf.exception.toString());
        return false;
    }
    
    Serial.printf("Model loaded! Arena used: ~%d bytes\n", ARENA_SIZE);
    return true;
}


// CAMERA INITIALIZATION
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
    config.frame_size = FRAMESIZE_QVGA;  // 320x240
    config.fb_count = 1;
    config.grab_mode = CAMERA_GRAB_LATEST;
    
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init error: 0x%x\n", err);
        return false;
    }
    
    return true;
}


// TILE CLASSIFICATION
/*
 * Classify a single tile from the full image.
 * 
 * The tile indexing follows this format:
 *   - Index 0 = bottom-right corner
 *   - Columns go right-to-left
 *   - Rows go bottom-to-top within each column
 * 
 * Parameters:
 *   rgb565Data - Pointer to full image data (RGB565 format)
 *   imgWidth, imgHeight - Full image dimensions
 *   tileCol - Column in 8x8 grid (0-7, where 0=rightmost)
 *   tileRow - Row in 8x8 grid (0-7, where 0=bottom)
 * 
 * Returns: 0 (Empty), 1 (Red), or 2 (Blue)
 */
int classifyTile(uint16_t* rgb565Data, int imgWidth, int imgHeight, 
                 int tileCol, int tileRow) {
    
    if (!modelLoaded) return 0;
    
    // Calculate tile boundaries in original image coordinates
    // tileCol: 0 = rightmost (image col 7), 7 = leftmost (image col 0)
    // tileRow: 0 = bottom (image row 7), 7 = top (image row 0)
    
    int imgCol = 7 - tileCol;  // Convert to image column (0=left, 7=right)
    int imgRow = 7 - tileRow;  // Convert to image row (0=top, 7=bottom)
    
    float tileWidth = imgWidth / 8.0f;
    float tileHeight = imgHeight / 8.0f;
    
    int left = (int)(imgCol * tileWidth);
    int top = (int)(imgRow * tileHeight);
    int right = (int)((imgCol + 1) * tileWidth);
    int bottom = (int)((imgRow + 1) * tileHeight);
    
    // Extract and resize tile to TILE_SIZE x TILE_SIZE
    // Simple nearest-neighbor resize
    int srcTileW = right - left;
    int srcTileH = bottom - top;
    
    int idx = 0;
    for (int y = 0; y < TILE_SIZE; y++) {
        int srcY = top + (y * srcTileH / TILE_SIZE);
        for (int x = 0; x < TILE_SIZE; x++) {
            int srcX = left + (x * srcTileW / TILE_SIZE);
            
            // Get pixel from source image
            uint16_t pixel = rgb565Data[srcY * imgWidth + srcX];
            
            // Convert RGB565 to normalized float [0, 1]
            float r = ((pixel >> 11) & 0x1F) / 31.0f;
            float g = ((pixel >> 5) & 0x3F) / 63.0f;
            float b = (pixel & 0x1F) / 31.0f;
            
            tileBuffer[idx++] = r;
            tileBuffer[idx++] = g;
            tileBuffer[idx++] = b;
        }
    }
    
    // Run inference on tile
    if (!tf.predict(tileBuffer).isOk()) {
        Serial.printf("Tile (%d,%d) inference failed\n", tileCol, tileRow);
        return 0;
    }
    
    // Get predicted class (argmax)
    float maxProb = tf.output(0);
    int predicted = 0;
    
    for (int c = 1; c < 3; c++) {
        if (tf.output(c) > maxProb) {
            maxProb = tf.output(c);
            predicted = c;
        }
    }
    
    return predicted;
}

/*
 * Recognize the full board by classifying all 64 tiles.
 * Populates recognizedState[64] with the predictions.
 */
bool recognizeBoardFromTiles() {
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
    
    uint16_t* rgb565 = (uint16_t*)fb->buf;
    int imgWidth = fb->width;
    int imgHeight = fb->height;
    
    Serial.println("Classifying 64 tiles...");
    unsigned long startTime = millis();
    
    // Classify all 64 tiles
    // Tile index follows: columns right->left, rows bottom->top
    // Index 0 = (col=0, row=0) = bottom-right
    int tileIndex = 0;
    for (int col = 0; col < 8; col++) {       // 0 = rightmost
        for (int row = 0; row < 8; row++) {   // 0 = bottom
            int predicted = classifyTile(rgb565, imgWidth, imgHeight, col, row);
            recognizedState[tileIndex] = predicted;
            tileIndex++;
        }
    }
    
    esp_camera_fb_return(fb);
    
    unsigned long elapsed = millis() - startTime;
    Serial.printf("All tiles classified in %lu ms (avg %.1f ms/tile)\n", 
                  elapsed, elapsed / 64.0f);
    
    // Print recognized board
    Serial.println("\nRecognized board:");
    printBoardArray(recognizedState);
    
    return true;
}


// HUMAN TURN

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
                    Serial.println("\n HUMAN WINS! \n");
                } else if (isBoardFullState(gameState)) {
                    gameOver = true;
                    winner = 0;
                    Serial.println("\n DRAW! \n");
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

// HELPER MACROS

#ifndef max
#define max(a,b) ((a)>(b)?(a):(b))
#endif

#ifndef min
#define min(a,b) ((a)<(b)?(a):(b))
#endif

// STATE SANITIZER (Fixes Camera Glitches)
// Ex: if the camera sees a piece in row 3 but empty in row 2, gravity is broken.
// This function fills the gaps so the AI doesn't crash or make illegal moves.
void sanitizeState(int8_t* state) {
    for (int col = 0; col < COLS; col++) {
        bool foundPiece = false;
        // Scan from top (Row 7) down to bottom (Row 0)
        for (int row = ROWS - 1; row >= 0; row--) {
            int8_t cell = state[INDEX(col, row)];
            
            if (cell != EMPTY) {
                foundPiece = true;
            } else if (foundPiece) {
                // Gap found below a piece! Camera glitch.
                // Assume it's the opponent to play defensively.
                state[INDEX(col, row)] = HUMAN; 
            }
        }
    }
}


// AI TURN (Camera for Prediction Only)

void handleAITurn() {
    //thinkingAnimation();
    
    // We create a temporary board for the AI to "think" with.
    int8_t stateForAI[64];
    bool usingCamera = false;
    
    // ACQUIRE STATE FROM CAMERA
    if (cameraReady && modelLoaded) {
        Serial.println("\n[Vision] Scanning board for prediction...");
        
        if (recognizeBoardFromTiles()) {
            // Load what the camera sees into the temporary AI buffer
            memcpy(stateForAI, recognizedState, 64);
            usingCamera = true;
            Serial.println("[Vision] Prediction based on CAMERA image.");
        } else {
            // If camera failed, use internal memory.
            memcpy(stateForAI, gameState, 64);
            Serial.println("[Vision] Failed. Prediction based on MEMORY.");
        }
    } else {
        memcpy(stateForAI, gameState, 64);
        Serial.println("[Vision] Camera disabled. Prediction based on MEMORY.");
    }
    
    // SANITIZE (Fix Gravity Glitches)
    if (usingCamera) {
        // We still fix floating pieces so the Minimax math doesn't break
        sanitizeState(stateForAI);
    }


    // CALCULATE MOVE (Using stateForAI)
    Serial.println("\n[AI] Thinking...");
    // printBoardArray(stateForAI); // view what the AI sees
    
    unsigned long startTime = millis();
    int aiCol = findBestMoveForState(stateForAI);
    unsigned long duration = millis() - startTime;
    
    Serial.printf("[AI] Chose column %d (Time: %lu ms)\n", aiCol, duration);
    
    // EXECUTE MOVE (Apply to Official gameState)
    // We attempt to drop the piece into the OFFICIAL game state.
    int row = findLowestEmptyRow(gameState, aiCol);
    
    if (row != -1) {
        // Update the official source of truth
        gameState[INDEX(aiCol, row)] = AI;
        
        animateDrop(aiCol, row, AI);
        printBoardArray(gameState);
        
        if (checkWinInState(gameState, AI)) {
            gameOver = true;
            winner = AI;
            Serial.println("\n AI WINS! \n");
        } else if (isBoardFullState(gameState)) {
            gameOver = true;
            winner = 0;
            Serial.println("\n DRAW! \n");
        } else {
            isHumanTurn = true;
            Serial.println("Your turn! Press button 0-7\n");
        }
    } else {
        // Fallback: The camera might have seen an open spot in 'stateForAI',
        // but the internal 'gameState' knows that column is actually full.
        Serial.printf("[Mismatch] Camera thought col %d was open, but memory says full.\n", aiCol);
        
        // Find the first valid column based on internal memory
        for (int c = 0; c < COLS; c++) {
            row = findLowestEmptyRow(gameState, c);
            if (row != -1) {
                gameState[INDEX(c, row)] = AI;
                animateDrop(c, row, AI);
                isHumanTurn = true;
                break;
            }
        }
    }
}

// AI - MINIMAX MODEL

// Weight board positions: Center columns and higher rows are more valuable
const int positionWeights[8][8] = {
    {3, 4,  5,  7,  7,  5,  4, 3}, // Row 0 (Bottom)
    {4, 6,  8, 10, 10,  8,  6, 4}, // Row 1
    {5, 8, 11, 13, 13, 11,  8, 5}, // Row 2
    {5, 8, 11, 13, 13, 11,  8, 5}, // Row 3
    {4, 6,  8, 10, 10,  8,  6, 4}, // Row 4
    {3, 4,  5,  7,  7,  5,  4, 3}, // Row 5
    {2, 3,  4,  5,  5,  4,  3, 2}, // Row 6
    {1, 2,  3,  4,  4,  3,  2, 1}  // Row 7 (Top)
};

int scorePosition(int8_t* state, int8_t player) {
    int score = 0;
    for (int col = 0; col < COLS; col++) {
        for (int row = 0; row < ROWS; row++) {
            if (state[INDEX(col, row)] == player) {
                score += positionWeights[row][col];
            }
        }
    }
    return score;
}

int findBestMoveForState(int8_t* state) {
    // Can I win right now?
    for (int col = 0; col < COLS; col++) {
        int row = findLowestEmptyRow(state, col);
        if (row != -1) {
            state[INDEX(col, row)] = AI;
            if (checkWinInState(state, AI)) {
                state[INDEX(col, row)] = EMPTY;
                Serial.printf(">> Immediate Win found at col %d\n", col);
                return col;
            }
            state[INDEX(col, row)] = EMPTY;
        }
    }

    // Must I block a human win?
    for (int col = 0; col < COLS; col++) {
        int row = findLowestEmptyRow(state, col);
        if (row != -1) {
            state[INDEX(col, row)] = HUMAN;
            if (checkWinInState(state, HUMAN)) {
                state[INDEX(col, row)] = EMPTY;
                Serial.printf(">> Blocking Human win at col %d\n", col);
                return col;
            }
            state[INDEX(col, row)] = EMPTY;
        }
    }

    // Minimax Search
    int bestScore = -10000000;
    int bestCol = -1;
    
    // Search center columns first to maximize pruning efficiency
    int colOrder[] = {3, 4, 2, 5, 1, 6, 0, 7};
    
    // Use the defined depth
    int effectiveDepth = AI_DEPTH; 
    
    for (int i = 0; i < COLS; i++) {
        int col = colOrder[i];
        int row = findLowestEmptyRow(state, col);
        
        if (row != -1) {
            state[INDEX(col, row)] = AI;
            
            // Initial recursive call
            int score = minimaxState(state, effectiveDepth - 1, -100000000, 100000000, false);
            
            state[INDEX(col, row)] = EMPTY;

            if (score > bestScore) {
                bestScore = score;
                bestCol = col;
            }
        }
    }
    
    // Fallback
    if (bestCol == -1) {
        for (int col = 0; col < COLS; col++) {
            if (findLowestEmptyRow(state, col) != -1) return col;
        }
    }
    
    return bestCol;
}

int minimaxState(int8_t* state, int depth, int alpha, int beta, bool maximizing) {
    // Check terminal states
    if (checkWinInState(state, AI)) return 1000000 + depth; // Win sooner is better
    if (checkWinInState(state, HUMAN)) return -1000000 - depth; // Lose later is better
    if (isBoardFullState(state)) return 0;
    if (depth == 0) return evaluateState(state);
    
    int colOrder[] = {3, 4, 2, 5, 1, 6, 0, 7};
    
    if (maximizing) {
        int maxScore = -100000000;
        for (int i = 0; i < COLS; i++) {
            int col = colOrder[i];
            int row = findLowestEmptyRow(state, col);
            if (row == -1) continue;
            
            state[INDEX(col, row)] = AI;
            int score = minimaxState(state, depth - 1, alpha, beta, false);
            state[INDEX(col, row)] = EMPTY;
            
            maxScore = max(maxScore, score);
            alpha = max(alpha, score);
            if (beta <= alpha) break; // Beta cutoff
        }
        return maxScore;
    } else {
        int minScore = 100000000;
        for (int i = 0; i < COLS; i++) {
            int col = colOrder[i];
            int row = findLowestEmptyRow(state, col);
            if (row == -1) continue;
            
            state[INDEX(col, row)] = HUMAN;
            int score = minimaxState(state, depth - 1, alpha, beta, true);
            state[INDEX(col, row)] = EMPTY;
            
            minScore = min(minScore, score);
            beta = min(beta, score);
            if (beta <= alpha) break; // Alpha cutoff
        }
        return minScore;
    }
}

int evaluateState(int8_t* state) {
    int score = 0;
    
    // Material score based on board position (Center control)
    score += scorePosition(state, AI);
    score -= scorePosition(state, HUMAN);
    
    // Window Evaluation (Horizontal, Vertical, Diagonal)
    // Horizontal
    for (int row = 0; row < ROWS; row++) {
        for (int col = 0; col < COLS - 3; col++) {
            score += evalWindow(state[INDEX(col,row)], state[INDEX(col+1,row)],
                               state[INDEX(col+2,row)], state[INDEX(col+3,row)]);
        }
    }
    // Vertical
    for (int col = 0; col < COLS; col++) {
        for (int row = 0; row < ROWS - 3; row++) {
            score += evalWindow(state[INDEX(col,row)], state[INDEX(col,row+1)],
                               state[INDEX(col,row+2)], state[INDEX(col,row+3)]);
        }
    }
    // Diagonal /
    for (int col = 0; col < COLS - 3; col++) {
        for (int row = 0; row < ROWS - 3; row++) {
            score += evalWindow(state[INDEX(col,row)], state[INDEX(col+1,row+1)],
                               state[INDEX(col+2,row+2)], state[INDEX(col+3,row+3)]);
        }
    }
    // Diagonal \ 
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
    
    // Tuned weights for better aggression/defense
    if (aiCnt == 4) return 10000;
    if (aiCnt == 3 && emptyCnt == 1) return 100;
    if (aiCnt == 2 && emptyCnt == 2) return 5;
    
    // Block opponent! Higher weight to blocking 3 than creating 2
    if (humanCnt == 3 && emptyCnt == 1) return -90; 
    if (humanCnt == 2 && emptyCnt == 2) return -10;
    
    return 0;
}

// GAME STATE HELPERS

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


// BUTTON INPUT

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


// LED DISPLAY

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


// DEBUG

void printBoardArray(int8_t* state) {
    char symbols[] = {'.', 'R', 'B'};
    
    Serial.println("\nBoard (visual, row 7=top):");
    Serial.println("  0 1 2 3 4 5 6 7  (columns)");
    
    for (int row = ROWS - 1; row >= 0; row--) {
        Serial.printf("%d ", row);
        for (int col = 0; col < COLS; col++) {
            int8_t cell = state[INDEX(col, row)];
            Serial.printf("%c ", symbols[cell]);
        }
        Serial.println();
    }
    
    // Also print raw 64-element array
    Serial.print("\nRaw array [64]: ");
    for (int i = 0; i < 64; i++) {
        Serial.print(state[i]);
        if (i < 63) Serial.print(",");
    }
    Serial.println("\n");
}

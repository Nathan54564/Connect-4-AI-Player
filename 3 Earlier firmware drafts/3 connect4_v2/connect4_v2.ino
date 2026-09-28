/*
 * Connect 4 AI with Camera
 * =========================
 * 
 * This version:
 * - Uses camera to capture images
 * - Processes image with simple color detection (no TFLite needed)
 * - Uses Minimax AI to decide moves
 * - Maintains game state array as source of truth
 * 
 * Hardware:
 * - XIAO ESP32-S3 Sense (with OV2640 camera)
 * - 8x8 NeoPixel LED Matrix on D0
 * - 8 Buttons (voltage divider) on D1
 * 
 * Required Libraries:
 * - Adafruit NeoPixel
 */

#include <Adafruit_NeoPixel.h>
#include "esp_camera.h"

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

// AI difficulty (search depth for Minimax)
#define AI_DEPTH 5

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
// HELPER MACROS
// ============================================================================

#define INDEX(col, row) ((row) * 8 + (col))

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

Adafruit_NeoPixel pixels(NUMPIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

// Colors
uint32_t COLOR_RED;
uint32_t COLOR_BLUE;
uint32_t COLOR_OFF;
uint32_t COLOR_YELLOW;
uint32_t COLOR_GREEN;

// Game state - THE SOURCE OF TRUTH
int8_t gameState[64];           // Current game state
int8_t recognizedState[64];     // State recognized from camera

// Game status
bool isHumanTurn = true;
bool gameOver = false;
int8_t winner = 0;

// Hardware status
bool cameraReady = false;
bool buttonHeld = false;

// ============================================================================
// FUNCTION DECLARATIONS (Arduino requires these)
// ============================================================================

void resetGame();
void displayBoard();
void startupAnimation();
void flashScreen(uint32_t color, int times);
void flashColumn(int col, uint32_t color, int times);
void animateDrop(int col, int targetRow, int8_t player);
void thinkingAnimation();
void showWinnerAnimation();
void setPixel(int col, int row, uint32_t color);
int pixelIndex(int col, int row);

bool initCamera();
bool recognizeBoardFromCamera();

int readButton();

void handleHumanTurn();
void handleAITurn();

int findLowestEmptyRow(int8_t* state, int col);
bool checkWinInState(int8_t* state, int8_t player);
bool isBoardFullState(int8_t* state);
int findBestMoveForState(int8_t* state);
int minimaxState(int8_t* state, int depth, int alpha, int beta, bool maximizing);
int evaluateState(int8_t* state);
int evalWindow(int8_t a, int8_t b, int8_t c, int8_t d);

void printBoardArray(int8_t* state);

// ============================================================================
// SETUP
// ============================================================================

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n============================================");
    Serial.println("  Connect 4 AI - Camera + Color Detection");
    Serial.println("============================================\n");
    
    // Initialize random
    randomSeed(analogRead(A0) + micros());
    
    // Initialize NeoPixels
    pixels.begin();
    pixels.setBrightness(30);
    
    COLOR_RED = pixels.Color(150, 0, 0);
    COLOR_BLUE = pixels.Color(0, 0, 150);
    COLOR_OFF = pixels.Color(0, 0, 0);
    COLOR_YELLOW = pixels.Color(100, 100, 0);
    COLOR_GREEN = pixels.Color(0, 100, 0);
    
    // Initialize buttons
    analogReadResolution(12);
    
    // Reset game
    resetGame();
    
    // Startup animation
    startupAnimation();
    
    // Initialize camera
    Serial.println("Initializing camera...");
    cameraReady = initCamera();
    
    if (cameraReady) {
        Serial.println("Camera ready!");
        flashScreen(COLOR_GREEN, 2);
    } else {
        Serial.println("Camera failed! Game still works without it.");
        flashScreen(COLOR_YELLOW, 3);
    }
    
    Serial.println("\nGame ready! Human (RED) plays first.");
    Serial.println("Press button 0-7 to drop a piece.\n");
    
    displayBoard();
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
    if (gameOver) {
        showWinnerAnimation();
        
        // Any button resets game
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
// CAMERA FUNCTIONS
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

/*
 * Recognize board state from camera using color detection.
 * This looks at the colors in the image to determine red/blue/empty.
 * 
 * Note: This is a simple approach. For better accuracy, you would
 * use a trained neural network (TFLite), but that requires a 
 * compatible library version.
 */
bool recognizeBoardFromCamera() {
    if (!cameraReady) {
        return false;
    }
    
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed!");
        return false;
    }
    
    Serial.printf("Captured: %dx%d\n", fb->width, fb->height);
    
    // Calculate grid cell size
    // Assuming the LED matrix takes up most of the image
    int cellWidth = fb->width / 8;
    int cellHeight = fb->height / 8;
    
    uint16_t* pixels_data = (uint16_t*)fb->buf;
    
    // Analyze each cell
    for (int gridRow = 0; gridRow < 8; gridRow++) {
        for (int gridCol = 0; gridCol < 8; gridCol++) {
            // Sample center of each cell
            int centerX = gridCol * cellWidth + cellWidth / 2;
            int centerY = gridRow * cellHeight + cellHeight / 2;
            
            // Average colors in a small region around center
            int totalR = 0, totalG = 0, totalB = 0;
            int sampleCount = 0;
            
            for (int dy = -2; dy <= 2; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    int x = centerX + dx;
                    int y = centerY + dy;
                    if (x >= 0 && x < fb->width && y >= 0 && y < fb->height) {
                        uint16_t pixel = pixels_data[y * fb->width + x];
                        
                        // Extract RGB from RGB565
                        int r = ((pixel >> 11) & 0x1F) << 3;
                        int g = ((pixel >> 5) & 0x3F) << 2;
                        int b = (pixel & 0x1F) << 3;
                        
                        totalR += r;
                        totalG += g;
                        totalB += b;
                        sampleCount++;
                    }
                }
            }
            
            int avgR = totalR / sampleCount;
            int avgG = totalG / sampleCount;
            int avgB = totalB / sampleCount;
            
            // Classify based on color
            // Note: Thresholds may need adjustment based on your setup
            int8_t detected = EMPTY;
            
            if (avgR > 100 && avgR > avgG * 2 && avgR > avgB * 2) {
                detected = HUMAN;  // Red
            } else if (avgB > 100 && avgB > avgG * 2 && avgB > avgR * 2) {
                detected = AI;  // Blue
            }
            
            // Map grid position to game state
            // Image row 0 = top = game row 7
            int gameRow = 7 - gridRow;
            recognizedState[INDEX(gridCol, gameRow)] = detected;
        }
    }
    
    esp_camera_fb_return(fb);
    
    Serial.println("Camera recognized board:");
    printBoardArray(recognizedState);
    
    return true;
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
                // Valid move - update game state
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
    //thinkingAnimation();
    
    // Create a copy of state for AI to analyze
    int8_t stateForAI[64];
    
    // Option 1: Try to use camera recognition
    if (cameraReady) {
        Serial.println("Capturing board with camera...");
        if (recognizeBoardFromCamera()) {
            // Use recognized state for AI decision
            memcpy(stateForAI, recognizedState, 64);
            Serial.println("Using camera-recognized state for AI.");
        } else {
            // Fall back to internal state
            memcpy(stateForAI, gameState, 64);
            Serial.println("Camera failed, using internal state.");
        }
    } else {
        // No camera, use internal state
        memcpy(stateForAI, gameState, 64);
        Serial.println("Using internal state for AI.");
    }
    
    // Find best move using Minimax
    Serial.println("AI calculating best move...");
    unsigned long startTime = millis();
    int aiCol = findBestMoveForState(stateForAI);
    unsigned long elapsed = millis() - startTime;
    
    Serial.printf("AI chose column %d (took %lu ms)\n", aiCol, elapsed);
    
    // Apply move to the ACTUAL game state
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
        // Column full - find any valid column
        Serial.printf("Column %d full! Finding alternative...\n", aiCol);
        for (int c = 0; c < COLS; c++) {
            row = findLowestEmptyRow(gameState, c);
            if (row != -1) {
                gameState[INDEX(c, row)] = AI;
                animateDrop(c, row, AI);
                Serial.printf("AI plays column %d instead\n", c);
                break;
            }
        }
        isHumanTurn = true;
    }
}

// ============================================================================
// AI - MINIMAX ALGORITHM
// ============================================================================

int findBestMoveForState(int8_t* state) {
    int bestScore = -1000000;
    int bestCol = -1;
    
    // Try columns center-first
    int colOrder[] = {3, 4, 2, 5, 1, 6, 0, 7};
    
    for (int i = 0; i < COLS; i++) {
        int col = colOrder[i];
        int row = findLowestEmptyRow(state, col);
        
        if (row != -1) {
            // Try move
            state[INDEX(col, row)] = AI;
            
            int score;
            if (checkWinInState(state, AI)) {
                score = 100000;
            } else {
                score = -minimaxState(state, AI_DEPTH - 1, -1000000, 1000000, false);
            }
            
            // Undo move
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
            if (findLowestEmptyRow(state, col) != -1) {
                return col;
            }
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
    
    // Center preference
    for (int row = 0; row < ROWS; row++) {
        if (state[INDEX(3, row)] == AI) score += 3;
        if (state[INDEX(4, row)] == AI) score += 3;
        if (state[INDEX(3, row)] == HUMAN) score -= 3;
        if (state[INDEX(4, row)] == HUMAN) score -= 3;
    }
    
    // Horizontal windows
    for (int row = 0; row < ROWS; row++) {
        for (int col = 0; col < COLS - 3; col++) {
            score += evalWindow(
                state[INDEX(col, row)], state[INDEX(col+1, row)],
                state[INDEX(col+2, row)], state[INDEX(col+3, row)]
            );
        }
    }
    
    // Vertical windows
    for (int col = 0; col < COLS; col++) {
        for (int row = 0; row < ROWS - 3; row++) {
            score += evalWindow(
                state[INDEX(col, row)], state[INDEX(col, row+1)],
                state[INDEX(col, row+2)], state[INDEX(col, row+3)]
            );
        }
    }
    
    // Diagonal / windows
    for (int col = 0; col < COLS - 3; col++) {
        for (int row = 0; row < ROWS - 3; row++) {
            score += evalWindow(
                state[INDEX(col, row)], state[INDEX(col+1, row+1)],
                state[INDEX(col+2, row+2)], state[INDEX(col+3, row+3)]
            );
        }
    }
    
    // Diagonal \ windows
    for (int col = 0; col < COLS - 3; col++) {
        for (int row = 3; row < ROWS; row++) {
            score += evalWindow(
                state[INDEX(col, row)], state[INDEX(col+1, row-1)],
                state[INDEX(col+2, row-2)], state[INDEX(col+3, row-3)]
            );
        }
    }
    
    return score;
}

int evalWindow(int8_t a, int8_t b, int8_t c, int8_t d) {
    int aiCnt = (a == AI) + (b == AI) + (c == AI) + (d == AI);
    int humanCnt = (a == HUMAN) + (b == HUMAN) + (c == HUMAN) + (d == HUMAN);
    int emptyCnt = (a == EMPTY) + (b == EMPTY) + (c == EMPTY) + (d == EMPTY);
    
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
        if (state[INDEX(col, row)] == EMPTY) {
            return row;
        }
    }
    return -1;
}

bool checkWinInState(int8_t* state, int8_t player) {
    // Horizontal
    for (int row = 0; row < ROWS; row++) {
        for (int col = 0; col < COLS - 3; col++) {
            if (state[INDEX(col, row)] == player &&
                state[INDEX(col+1, row)] == player &&
                state[INDEX(col+2, row)] == player &&
                state[INDEX(col+3, row)] == player) {
                return true;
            }
        }
    }
    
    // Vertical
    for (int col = 0; col < COLS; col++) {
        for (int row = 0; row < ROWS - 3; row++) {
            if (state[INDEX(col, row)] == player &&
                state[INDEX(col, row+1)] == player &&
                state[INDEX(col, row+2)] == player &&
                state[INDEX(col, row+3)] == player) {
                return true;
            }
        }
    }
    
    // Diagonal /
    for (int col = 0; col < COLS - 3; col++) {
        for (int row = 0; row < ROWS - 3; row++) {
            if (state[INDEX(col, row)] == player &&
                state[INDEX(col+1, row+1)] == player &&
                state[INDEX(col+2, row+2)] == player &&
                state[INDEX(col+3, row+3)] == player) {
                return true;
            }
        }
    }
    
    // Diagonal 
    for (int col = 0; col < COLS - 3; col++) {
        for (int row = 3; row < ROWS; row++) {
            if (state[INDEX(col, row)] == player &&
                state[INDEX(col+1, row-1)] == player &&
                state[INDEX(col+2, row-2)] == player &&
                state[INDEX(col+3, row-3)] == player) {
                return true;
            }
        }
    }
    
    return false;
}

bool isBoardFullState(int8_t* state) {
    for (int col = 0; col < COLS; col++) {
        if (findLowestEmptyRow(state, col) != -1) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// GAME MANAGEMENT
// ============================================================================

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
// LED DISPLAY
// ============================================================================

int pixelIndex(int col, int row) {
    // Zigzag wiring pattern
    if (row % 2 == 0) {
        return row * 8 + col;
    } else {
        return row * 8 + (7 - col);
    }
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
            if (cell == HUMAN) {
                setPixel(col, row, COLOR_RED);
            } else if (cell == AI) {
                setPixel(col, row, COLOR_BLUE);
            }
        }
    }
    
    pixels.show();
}

void animateDrop(int col, int targetRow, int8_t player) {
    uint32_t color = (player == HUMAN) ? COLOR_RED : COLOR_BLUE;
    
    // Temporarily remove piece for animation
    int8_t saved = gameState[INDEX(col, targetRow)];
    gameState[INDEX(col, targetRow)] = EMPTY;
    
    // Animate falling from top
    for (int row = ROWS - 1; row >= targetRow; row--) {
        displayBoard();
        setPixel(col, row, color);
        pixels.show();
        delay(50);
    }
    
    // Restore piece
    gameState[INDEX(col, targetRow)] = saved;
    displayBoard();
}

void flashColumn(int col, uint32_t color, int times) {
    for (int t = 0; t < times; t++) {
        for (int row = 0; row < ROWS; row++) {
            setPixel(col, row, color);
        }
        pixels.show();
        delay(100);
        displayBoard();
        delay(100);
    }
}

void flashScreen(uint32_t color, int times) {
    for (int t = 0; t < times; t++) {
        for (int i = 0; i < NUMPIXELS; i++) {
            pixels.setPixelColor(i, color);
        }
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
    static bool flashOn = false;
    
    if (millis() - lastFlash > 400) {
        lastFlash = millis();
        flashOn = !flashOn;
        
        uint32_t color;
        if (winner == HUMAN) {
            color = COLOR_RED;
        } else if (winner == AI) {
            color = COLOR_BLUE;
        } else {
            color = COLOR_YELLOW;
        }
        
        if (flashOn) {
            for (int i = 0; i < NUMPIXELS; i++) {
                pixels.setPixelColor(i, color);
            }
        } else {
            displayBoard();
        }
        pixels.show();
    }
}

void startupAnimation() {
    // Red sweep
    for (int i = 0; i < NUMPIXELS; i++) {
        pixels.setPixelColor(i, COLOR_RED);
        pixels.show();
        delay(8);
    }
    delay(100);
    
    // Blue sweep
    for (int i = 0; i < NUMPIXELS; i++) {
        pixels.setPixelColor(i, COLOR_BLUE);
        pixels.show();
        delay(8);
    }
    delay(100);
    
    // Clear
    pixels.clear();
    pixels.show();
    delay(200);
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
            int8_t cell = state[INDEX(col, row)];
            if (cell == HUMAN) c = 'R';
            else if (cell == AI) c = 'B';
            Serial.printf("%c ", c);
        }
        Serial.println();
    }
    Serial.println();
}

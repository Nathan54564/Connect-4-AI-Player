#include <Adafruit_NeoPixel.h>
#include "esp_camera.h"
#include <EloquentTinyML.h> // Library Version 2.4.2
#include "model_data.h"

// --- HARDWARE CONFIGURATION ---
#define LED_PIN     D0
#define BUTTON_PIN  D1
#define NUMPIXELS   64

// --- CAMERA PIN DEFINITIONS (XIAO ESP32S3 SENSE) ---
#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 10
#define SIOD_GPIO_NUM 40
#define SIOC_GPIO_NUM 39
#define Y9_GPIO_NUM 48
#define Y8_GPIO_NUM 11
#define Y7_GPIO_NUM 12
#define Y6_GPIO_NUM 14
#define Y5_GPIO_NUM 16
#define Y4_GPIO_NUM 18
#define Y3_GPIO_NUM 17
#define Y2_GPIO_NUM 15
#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM 47
#define PCLK_GPIO_NUM 13

Adafruit_NeoPixel pixels(NUMPIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

// --- AI SETTINGS (For Version 2.4.2) ---
#define N_INPUTS (128 * 128 * 3) 
#define N_OUTPUTS 3
#define TENSOR_ARENA_SIZE 100 * 1024 

// Define the interpreter using 2.4.2 Syntax
Eloquent::TinyML::TfLite<N_INPUTS, N_OUTPUTS, TENSOR_ARENA_SIZE> interpreter;

// Global input buffer (Required for 2.4.2 as it doesn't expose internal pointer)
float *inputBuffer = nullptr;

// --- GAME STATE ---
int board[8][8]; 
int columnHeights[8];
bool isRedTurn = true; // Red = Human, Blue = AI
bool isButtonHeld = false;

// --- PROTOTYPES ---
void setupCamera();
void runVisionInference();
int getBestMove(int depth);
int getColumnFromVoltage();
void setZigZagPixel(int col, int row, uint32_t color);
void drawBoard();
int scorePosition(int piece);
bool checkWin(int player);

void setup() {
  Serial.begin(115200);
  
  // 1. LED Setup
  pixels.begin();
  pixels.setBrightness(20);
  pixels.show();
  
  // 2. Button Setup
  analogReadResolution(12);
  
  // 3. Camera Setup
  setupCamera();

  // 4. Eloquens Setup
  // Allocate memory for input buffer in PSRAM (if available) or Heap
  inputBuffer = (float*) ps_malloc(N_INPUTS * sizeof(float));
  if (inputBuffer == NULL) {
    inputBuffer = (float*) malloc(N_INPUTS * sizeof(float));
  }

  // Initialize model (2.4.2 returns bool here)
  if (!interpreter.begin(connect4_vision_model)) {
    Serial.println("AI Model failed to initialize!");
    while (true) delay(1000);
  }

  // Initialize Game Logic Board
  memset(board, 0, sizeof(board));
  memset(columnHeights, 0, sizeof(columnHeights));
  
  Serial.println("System Ready. Human (Red) Start.");
}

void loop() {
  // --- STATE 1: HUMAN TURN (RED) ---
  if (isRedTurn) {
    int candidateColumn = getColumnFromVoltage();

    if (candidateColumn != -1) {
      delay(50); // Debounce
      int verifiedColumn = getColumnFromVoltage();

      if (candidateColumn == verifiedColumn && !isButtonHeld) {
        if (columnHeights[verifiedColumn] < 8) {
          int row = columnHeights[verifiedColumn];
          
          board[verifiedColumn][row] = 1; // 1 = Red
          columnHeights[verifiedColumn]++;
          drawBoard();
          
          isRedTurn = false; 
          Serial.println("Human moved. Analyzing board...");
          delay(2000); 
        }
        isButtonHeld = true;
      }
    } else {
      isButtonHeld = false;
    }
  }
  // --- STATE 2: VISION + AI TURN (BLUE) ---
  else {
    runVisionInference(); 
    
    Serial.println("AI Thinking...");
    int aiCol = getBestMove(4); // Depth 4
    
    if (aiCol != -1 && columnHeights[aiCol] < 8) {
      int row = columnHeights[aiCol];
      board[aiCol][row] = 2; // 2 = Blue
      columnHeights[aiCol]++;
      Serial.print("AI chooses column: ");
      Serial.println(aiCol);
    } else {
      Serial.println("AI Error: Board full or invalid move.");
    }
    
    drawBoard();
    isRedTurn = true; // Back to human
  }
  delay(10);
}

// --- VISION LOGIC (Adapted for 2.4.2) ---
void runVisionInference() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return;
  }

  // Use the global inputBuffer we allocated in setup
  int w = 128;
  int h = 128;
  int startX = (320 - 128) / 2;
  int startY = (240 - 128) / 2;
  
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      int camX = startX + x;
      int camY = startY + y;
      int index = (camY * 320 + camX) * 2; 
      
      uint16_t pixel = (fb->buf[index] << 8) | fb->buf[index + 1];
      // Normalize to 0.0 - 1.0 floats
      float r = ((pixel >> 11) & 0x1F) / 31.0;
      float g = ((pixel >> 5) & 0x3F) / 63.0;
      float b = (pixel & 0x1F) / 31.0;
      
      int tensor_index = (y * w + x) * 3;
      inputBuffer[tensor_index] = r;
      inputBuffer[tensor_index + 1] = g;
      inputBuffer[tensor_index + 2] = b;
    }
  }
  
  esp_camera_fb_return(fb);

  // Run inference using 2.4.2 syntax (pass the buffer)
  float* output = interpreter.predict(inputBuffer);
  
  Serial.println("Vision Result:");
  for (int col = 0; col < 8; col++) {
    columnHeights[col] = 0; 
    for (int row = 0; row < 8; row++) {
      int idx = col * 8 + row;
      
      float probEmpty = output[idx * 3 + 0];
      float probRed   = output[idx * 3 + 1];
      float probBlue  = output[idx * 3 + 2];
      
      int detectedClass = 0;
      if (probRed > probEmpty && probRed > probBlue) detectedClass = 1;
      else if (probBlue > probEmpty && probBlue > probRed) detectedClass = 2;
      
      board[col][row] = detectedClass;
      if (detectedClass != 0) columnHeights[col] = row + 1;
    }
  }
}

// --- GAME AI LOGIC (Cleaned & Checked) ---
int evaluateWindow(int window[], int piece) {
  int score = 0;
  int oppPiece = (piece == 1) ? 2 : 1;
  int countPiece = 0;
  int countEmpty = 0;
  int countOpp = 0;

  for (int i = 0; i < 4; i++) {
    if (window[i] == piece) countPiece++;
    else if (window[i] == 0) countEmpty++;
    else countOpp++;
  }

  if (countPiece == 4) score += 100;
  else if (countPiece == 3 && countEmpty == 1) score += 5;
  else if (countPiece == 2 && countEmpty == 2) score += 2;

  if (countOpp == 3 && countEmpty == 1) score -= 80;

  return score;
}

int scorePosition(int piece) {
  int score = 0;
  
  // Center Column Preference
  for(int r=0; r<8; r++) {
    if(board[3][r] == piece) score += 3;
    if(board[4][r] == piece) score += 3;
  }

  // Horizontal
  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 5; c++) {
      int window[4] = {board[c][r], board[c+1][r], board[c+2][r], board[c+3][r]};
      score += evaluateWindow(window, piece);
    }
  }

  // Vertical
  for (int c = 0; c < 8; c++) {
    for (int r = 0; r < 5; r++) {
      int window[4] = {board[c][r], board[c][r+1], board[c][r+2], board[c][r+3]};
      score += evaluateWindow(window, piece);
    }
  }

  // Diagonal /
  for (int r = 0; r < 5; r++) {
    for (int c = 0; c < 5; c++) {
      int window[4] = {board[c][r], board[c+1][r+1], board[c+2][r+2], board[c+3][r+3]};
      score += evaluateWindow(window, piece);
    }
  }

  // Diagonal \ 
  for (int r = 0; r < 5; r++) {
    for (int c = 3; c < 8; c++) {
      int window[4] = {board[c][r], board[c-1][r+1], board[c-2][r+2], board[c-3][r+3]};
      score += evaluateWindow(window, piece);
    }
  }

  return score;
}

bool checkWin(int player) {
  // Horizontal
  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 5; c++) {
      if (board[c][r] == player && board[c+1][r] == player && 
          board[c+2][r] == player && board[c+3][r] == player) return true;
    }
  }
  // Vertical
  for (int c = 0; c < 8; c++) {
    for (int r = 0; r < 5; r++) {
      if (board[c][r] == player && board[c][r+1] == player && 
          board[c][r+2] == player && board[c][r+3] == player) return true;
    }
  }
  // Diagonal /
  for (int c = 0; c < 5; c++) {
    for (int r = 0; r < 5; r++) {
      if (board[c][r] == player && board[c+1][r+1] == player && 
          board[c+2][r+2] == player && board[c+3][r+3] == player) return true;
    }
  }
  // Diagonal \ 
  for (int c = 3; c < 8; c++) {
    for (int r = 0; r < 5; r++) {
      if (board[c][r] == player && board[c-1][r+1] == player && 
          board[c-2][r+2] == player && board[c-3][r+3] == player) return true;
    }
  }
  return false;
}

int minimax(int depth, int alpha, int beta, bool maximizingPlayer) {
  if (checkWin(2)) return 100000;
  if (checkWin(1)) return -100000;
  if (depth == 0) return scorePosition(2);

  if (maximizingPlayer) {
    int maxEval = -999999;
    for (int c = 0; c < 8; c++) {
      if (columnHeights[c] < 8) {
        int r = columnHeights[c];
        board[c][r] = 2;
        columnHeights[c]++;
        int eval = minimax(depth - 1, alpha, beta, false);
        columnHeights[c]--;
        board[c][r] = 0;
        maxEval = max(maxEval, eval);
        alpha = max(alpha, eval);
        if (beta <= alpha) break;
      }
    }
    return maxEval;
  } else {
    int minEval = 999999;
    for (int c = 0; c < 8; c++) {
      if (columnHeights[c] < 8) {
        int r = columnHeights[c];
        board[c][r] = 1;
        columnHeights[c]++;
        int eval = minimax(depth - 1, alpha, beta, true);
        columnHeights[c]--;
        board[c][r] = 0;
        minEval = min(minEval, eval);
        beta = min(beta, eval);
        if (beta <= alpha) break;
      }
    }
    return minEval;
  }
}

int getBestMove(int depth) {
  int bestVal = -999999;
  int bestMove = -1;
  int order[] = {3, 4, 2, 5, 1, 6, 0, 7};

  for (int i = 0; i < 8; i++) {
    int c = order[i];
    if (columnHeights[c] < 8) {
      int r = columnHeights[c];
      
      board[c][r] = 2;
      columnHeights[c]++;
      
      int moveVal = minimax(depth - 1, -1000000, 1000000, false);
      
      columnHeights[c]--;
      board[c][r] = 0;
      
      if (moveVal > bestVal) {
        bestMove = c;
        bestVal = moveVal;
      }
    }
  }
  
  if (bestMove == -1) {
    for(int i=0; i<8; i++) if(columnHeights[i] < 8) return i;
  }
  return bestMove;
}

// --- HARDWARE HELPERS ---
void setupCamera() {
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
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.fb_location = CAMERA_FB_IN_PSRAM;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }
}

int getColumnFromVoltage() {
  pinMode(BUTTON_PIN, INPUT_PULLDOWN);
  delay(1);
  int reading = analogRead(BUTTON_PIN);
  if (reading > 3500)      { return 7; }
  else if (reading > 2900) { return 6; }
  else if (reading > 2500) { return 5; }
  else if (reading > 2100) { return 4; }
  else if (reading > 1600) { return 3; }
  else if (reading > 1100) { return 2; }
  else if (reading > 750)  { return 1; }
  else if (reading > 300)  { return 0; } 
  return -1;
}

void drawBoard() {
  pixels.clear(); 
  for (int col = 0; col < 8; col++) {
    for (int row = 0; row < 8; row++) {
      if (board[col][row] != 0) {
        uint32_t color = (board[col][row] == 1) ? pixels.Color(150, 0, 0) : pixels.Color(0, 0, 150);
        setZigZagPixel(col, row, color);
      }
    }
  }
  pixels.show();
}

void setZigZagPixel(int col, int row, uint32_t color) {
  int pixelIndex;
  if (row % 2 == 0) {
    pixelIndex = (row * 8) + col;
  } else {
    pixelIndex = (row * 8) + (7 - col);
  }
  pixels.setPixelColor(pixelIndex, color);
}
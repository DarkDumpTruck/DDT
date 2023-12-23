#pragma once

#include "core/util/common.h"
#include "core/util/xxhash64.h"

namespace Quoridor {

#ifdef QUORIDOR_WIDTH
constexpr const int WIDTH = QUORIDOR_WIDTH;
#else
constexpr const int WIDTH = 9;
#endif
#ifdef QUORIDOR_HEIGHT
constexpr const int HEIGHT = QUORIDOR_HEIGHT;
#else
constexpr const int HEIGHT = 9;
#endif

constexpr const int NUM_ACTIONS = 4 + 2 * (WIDTH - 1) * (HEIGHT - 1);
constexpr const int NUM_PLAYERS = 2;
constexpr const int NUM_WALLS = 10;
constexpr const int NUM_SYMMETRIES = 2;
constexpr const int PLAYER_0_INIT_Y = HEIGHT / 2;
constexpr const int PLAYER_1_INIT_Y = HEIGHT / 2;
constexpr const std::array<int, 3> BOARD_SHAPE = {2, WIDTH, HEIGHT};
constexpr const std::array<int, 3> CANONICAL_SHAPE = {8, WIDTH, HEIGHT};

using ActionType = uint8_t;

constexpr const ActionType MOVE_UP = 0;
constexpr const ActionType MOVE_DOWN = 1;
constexpr const ActionType MOVE_LEFT = 2;
constexpr const ActionType MOVE_RIGHT = 3;
constexpr const ActionType MOVE_PASS = NUM_ACTIONS;
constexpr const ActionType MOVE_ILLEGAL = NUM_ACTIONS + 1;

// reflect a coord on board to a number
int pos(int x, int y) { return x * HEIGHT + y; }

// reflect a vertical wall to a number, used in moves
int v_wall(int x, int y) { return 4 + x * (HEIGHT - 1) + y; }

// reflect a horizontal wall to a number, used in moves
int h_wall(int x, int y) {
  return 4 + (HEIGHT - 1) * (WIDTH - 1) + x * (HEIGHT - 1) + y;
}

// the wall number of (x,y) when look from the opponent side
int v_wall_op(int x, int y) { return v_wall(WIDTH - 2 - x, y); }
int h_wall_op(int x, int y) { return h_wall(WIDTH - 2 - x, y); }

// the wall number of (x,y) when look from player i
int v_wall(int x, int y, int i) { return i ? v_wall_op(x, y) : v_wall(x, y); }
int h_wall(int x, int y, int i) { return i ? h_wall_op(x, y) : h_wall(x, y); }

std::string action_to_string(const ActionType action, const bool player = 0) {
  if (action == MOVE_UP) return "Up";
  if (action == MOVE_DOWN) return "Down";
  if (action == MOVE_LEFT) return player ? "Right" : "Left";
  if (action == MOVE_RIGHT) return player ? "Left" : "Right";
  if (action == MOVE_PASS) return "Pass";
  char ret[4];
  if (action < (WIDTH - 1) * (HEIGHT - 1) + 4) {
    int x = (action - 4) / (HEIGHT - 1), y = (action - 4) % (HEIGHT - 1);
    if (player) x = WIDTH - 2 - x;
    ret[0] = 'a' + x;
    ret[1] = '0' + y + 1;
    ret[2] = '0' + y + 2;
  } else {
    int x = (action - 4 - (WIDTH - 1) * (HEIGHT - 1)) / (HEIGHT - 1),
        y = (action - 4 - (WIDTH - 1) * (HEIGHT - 1)) % (HEIGHT - 1);
    if (player) x = WIDTH - 2 - x;
    ret[0] = 'a' + WIDTH - 1 + y;
    ret[1] = '0' + x + 1;
    ret[2] = '0' + x + 2;
  }
  ret[3] = '\0';
  return ret;
}

ActionType string_to_action(const std::string &action, const bool player = 0) {
  if (action == "Up" || action == "u" || action == "U") return MOVE_UP;
  if (action == "Down" || action == "d" || action == "D") return MOVE_DOWN;
  if (action == "Left" || action == "l" || action == "L")
    return player ? MOVE_RIGHT : MOVE_LEFT;
  if (action == "Right" || action == "r" || action == "R")
    return player ? MOVE_LEFT : MOVE_RIGHT;
  if (action == "Pass" || action == "pass") return MOVE_PASS;
  if (action.size() != 3) return MOVE_ILLEGAL;
  if (action[0] - 'a' < WIDTH - 1) {
    uint8_t x = action[0] - 'a', y = action[1] - '0' - 1;
    if (player) x = WIDTH - 2 - x;
    return 4 + x * (HEIGHT - 1) + y;
  } else {
    uint8_t y = action[0] - 'a' - (WIDTH - 1), x = action[1] - '0' - 1;
    if (player) x = WIDTH - 2 - x;
    return 4 + (WIDTH - 1 + x) * (HEIGHT - 1) + y;
  }
}

using ActionType = Quoridor::ActionType;

// Basic storage of game board. Do not init itself when created.
class BoardTensor {
 public:
  const int Size = BOARD_SHAPE[0] * BOARD_SHAPE[1] * BOARD_SHAPE[2];
  BoardTensor() { storage_ = new uint8_t[Size]; }
  ~BoardTensor() { delete[] storage_; }
  BoardTensor(const BoardTensor &b) {
    storage_ = new uint8_t[Size];
    memcpy(storage_, b.storage_, Size);
  }
  uint8_t &operator()(int x, int y, int z) {
    assert(x < BOARD_SHAPE[0]);
    assert(y < BOARD_SHAPE[1]);
    assert(z < BOARD_SHAPE[2]);
    return storage_[x * BOARD_SHAPE[1] * BOARD_SHAPE[2] + y * BOARD_SHAPE[2] +
                    z];
  }
  const uint8_t &operator()(int x, int y, int z) const {
    return storage_[x * BOARD_SHAPE[1] * BOARD_SHAPE[2] + y * BOARD_SHAPE[2] +
                    z];
  }
  void set(int x, int y, int z, uint8_t val) const {
    storage_[x * BOARD_SHAPE[1] * BOARD_SHAPE[2] + y * BOARD_SHAPE[2] + z] =
        val;
  }
  bool operator==(const BoardTensor &board_) const noexcept {
    return storage_ == board_.storage_;
  }
  uint8_t &player0_x() {
    return storage_[(BOARD_SHAPE[1] - 1) * BOARD_SHAPE[2]];
  }
  const uint8_t &player0_x() const {
    return storage_[(BOARD_SHAPE[1] - 1) * BOARD_SHAPE[2]];
  }
  uint8_t &player1_x() {
    return storage_[(BOARD_SHAPE[1] - 1) * BOARD_SHAPE[2] + 1];
  }
  const uint8_t &player1_x() const {
    return storage_[(BOARD_SHAPE[1] - 1) * BOARD_SHAPE[2] + 1];
  }
  uint8_t &player0_y() {
    return storage_[(BOARD_SHAPE[1] - 1) * BOARD_SHAPE[2] + 2];
  }
  const uint8_t &player0_y() const {
    return storage_[(BOARD_SHAPE[1] - 1) * BOARD_SHAPE[2] + 2];
  }
  uint8_t &player1_y() {
    return storage_[(BOARD_SHAPE[1] - 1) * BOARD_SHAPE[2] + 3];
  }
  const uint8_t &player1_y() const {
    return storage_[(BOARD_SHAPE[1] - 1) * BOARD_SHAPE[2] + 3];
  }
  uint8_t &player0_cnt() {
    return storage_[(BOARD_SHAPE[1] - 1) * BOARD_SHAPE[2] + 4];
  }
  const uint8_t &player0_cnt() const {
    return storage_[(BOARD_SHAPE[1] - 1) * BOARD_SHAPE[2] + 4];
  }
  uint8_t &player1_cnt() {
    return storage_[(BOARD_SHAPE[1] - 1) * BOARD_SHAPE[2] + 5];
  }
  const uint8_t &player1_cnt() const {
    return storage_[(BOARD_SHAPE[1] - 1) * BOARD_SHAPE[2] + 5];
  }
  uint8_t &player_x(uint8_t player_id) {
    return storage_[(BOARD_SHAPE[1] - 1) * BOARD_SHAPE[2] + player_id];
  }
  const uint8_t &player_x(uint8_t player_id) const {
    return storage_[(BOARD_SHAPE[1] - 1) * BOARD_SHAPE[2] + player_id];
  }
  uint8_t &player_y(uint8_t player_id) {
    return storage_[(BOARD_SHAPE[1] - 1) * BOARD_SHAPE[2] + player_id + 2];
  }
  const uint8_t &player_y(uint8_t player_id) const {
    return storage_[(BOARD_SHAPE[1] - 1) * BOARD_SHAPE[2] + player_id + 2];
  }
  uint8_t &player_cnt(uint8_t player_id) {
    return storage_[(BOARD_SHAPE[1] - 1) * BOARD_SHAPE[2] + player_id + 4];
  }
  const uint8_t &player_cnt(uint8_t player_id) const {
    return storage_[(BOARD_SHAPE[1] - 1) * BOARD_SHAPE[2] + player_id + 4];
  }
  void set_zero() { std::fill(storage_, storage_ + Size, 0); }
  void set_constant(uint8_t val) { std::fill(storage_, storage_ + Size, val); }
  uint64_t hash() const noexcept { return XXHash64::hash(storage_, Size, 0); }

 private:
  uint8_t *storage_;
};

class GameState {
 private:
  /*
    The current game board, where board[0] represents the vertical walls, and
    board[1] the horizontal walls.
    Size of board[0] is (WIDTH-1) * HEIGHT.
    Size of board[1] is WIDTH * (HEIGHT-1).
  */
  BoardTensor board;

  /*
    The current valid places to place a wall for any player, not taking the
    *must have a path* rule into account.
  */
  BoardTensor valid_moves;

  /*
    The current player id.
  */
  bool current_player;

 public:
  GameState() { Init(); }
  GameState(BoardTensor board_, bool current_player_)
      : board(board_), current_player(current_player_) {}
  GameState(BoardTensor &&board_, bool current_player_)
      : board(board_), current_player(current_player_) {}

  void Init() {
    board.set_zero();
    valid_moves.set_constant(1);

    // board.player0_x() = 0;
    board.player0_y() = PLAYER_0_INIT_Y;
    board.player1_x() = WIDTH - 1;
    board.player1_y() = PLAYER_1_INIT_Y;
    board.player0_cnt() = NUM_WALLS;
    board.player1_cnt() = NUM_WALLS;

    current_player = 0;
  }

  std::unique_ptr<GameState> Copy() const {
    return std::make_unique<GameState>(*this);
  }

  uint64_t Hash() const noexcept {
    return board.hash() ^ (uint64_t)current_player;
  }

  bool Current_player() { return current_player; }

  int Num_actions() { return NUM_ACTIONS; }

  std::vector<uint8_t> Valid_moves() const {
    auto valids = std::vector<uint8_t>(NUM_ACTIONS, 0);

    // first four moves represents up, down, left, right
    valids[MOVE_UP] = board.player_y(current_player) &&
                      !board(1, board.player_x(current_player),
                             board.player_y(current_player) - 1);
    valids[MOVE_DOWN] = board.player_y(current_player) < HEIGHT - 1 &&
                        !board(1, board.player_x(current_player),
                               board.player_y(current_player));
    valids[MOVE_LEFT] =
        board.player_x(current_player) &&
        board.player_x(current_player) < WIDTH - 1 &&
        !board(0, board.player_x(current_player) - (!current_player),
               board.player_y(current_player));
    valids[MOVE_RIGHT] =
        !board(0, board.player_x(current_player) - current_player,
               board.player_y(current_player));

    // other moves represents validation of adding walls.
    if (board.player_cnt(current_player) == 0) {
      // if there's no valid wall left, do nothing.
    } else {
      for (int i = 0; i < WIDTH - 1; i++) {
        for (int j = 0; j < HEIGHT - 1; j++) {
          if (valid_moves(0, i, j)) {
            board.set(0, i, j, 1);
            board.set(0, i, j + 1, 1);
            valids[v_wall(i, j, current_player)] = checkPathExist();
            board.set(0, i, j, 0);
            board.set(0, i, j + 1, 0);
          }

          if (valid_moves(1, i, j)) {
            board.set(1, i, j, 1);
            board.set(1, i + 1, j, 1);
            valids[h_wall(i, j, current_player)] = checkPathExist();
            board.set(1, i, j, 0);
            board.set(1, i + 1, j, 0);
          }
        }
      }
    }

    return valids;
  }

  void Move(ActionType action) {
    if (action == MOVE_PASS) {
      current_player = !current_player;
      return;
    }

    assert(action != MOVE_ILLEGAL);

    if (action == MOVE_UP) {
      if (current_player == 0) {
        board.player0_y()--;
        if (board.player0_y() == board.player1_y() &&
            board.player0_x() == board.player1_x()) {
          if (board.player0_y() > 0 &&
              !board(1, board.player0_x(), board.player0_y() - 1)) {
            board.player0_y()--;
          } else if (board.player0_x() > 0 && board.player0_x() < WIDTH - 1 &&
                     !board(0, board.player0_x(), board.player0_y()) &&
                     !board(0, board.player0_x() - 1, board.player0_y())) {
            int dis_0 = shortestDistance(board.player0_x() - 1,
                                         board.player0_y(), WIDTH - 1);
            int dis_1 = shortestDistance(board.player0_x() + 1,
                                         board.player0_y(), WIDTH - 1);
            if (dis_0 < dis_1) {
              board.player0_x()--;
            } else {
              board.player0_x()++;
            }
          } else if (board.player0_x() < WIDTH - 1 &&
                     !board(0, board.player0_x(), board.player0_y())) {
            board.player0_x()++;
          } else if (board.player0_x() > 0 &&
                     !board(0, board.player0_x() - 1, board.player0_y())) {
            board.player0_x()--;
          } else {
            board.player0_y()++;
          }
        }
      } else {
        board.player1_y()--;
        if (board.player0_y() == board.player1_y() &&
            board.player0_x() == board.player1_x()) {
          if (board.player1_y() > 0 &&
              !board(1, board.player1_x(), board.player1_y() - 1)) {
            board.player1_y()--;
          } else if (board.player1_x() > 0 && board.player1_x() < WIDTH - 1 &&
                     !board(0, board.player1_x() - 1, board.player1_y()) &&
                     !board(0, board.player1_x(), board.player1_y())) {
            int dis_0 =
                shortestDistance(board.player1_x() - 1, board.player1_y(), 0);
            int dis_1 =
                shortestDistance(board.player1_x() + 1, board.player1_y(), 0);
            if (dis_0 <= dis_1) {
              board.player1_x()--;
            } else {
              board.player1_x()++;
            }
          } else if (board.player1_x() > 0 &&
                     !board(0, board.player1_x() - 1, board.player1_y())) {
            board.player1_x()--;
          } else if (board.player1_x() < WIDTH - 1 &&
                     !board(0, board.player1_x(), board.player1_y())) {
            board.player1_x()++;
          } else {
            board.player1_y()++;
          }
        }
      }
    } else if (action == MOVE_DOWN) {
      if (current_player == 0) {
        board.player0_y()++;
        if (board.player0_y() == board.player1_y() &&
            board.player0_x() == board.player1_x()) {
          if (board.player0_y() < HEIGHT - 1 &&
              !board(1, board.player0_x(), board.player0_y())) {
            board.player0_y()++;
          } else if (board.player0_x() > 0 && board.player0_x() < WIDTH - 1 &&
                     !board(0, board.player0_x(), board.player0_y()) &&
                     !board(0, board.player0_x() - 1, board.player0_y())) {
            int dis_0 = shortestDistance(board.player0_x() - 1,
                                         board.player0_y(), WIDTH - 1);
            int dis_1 = shortestDistance(board.player0_x() + 1,
                                         board.player0_y(), WIDTH - 1);
            if (dis_0 < dis_1) {
              board.player0_x()--;
            } else {
              board.player0_x()++;
            }
          } else if (board.player0_x() < WIDTH - 1 &&
                     !board(0, board.player0_x(), board.player0_y())) {
            board.player0_x()++;
          } else if (board.player0_x() > 0 &&
                     !board(0, board.player0_x() - 1, board.player0_y())) {
            board.player0_x()--;
          } else {
            board.player0_y()--;
          }
        }
      } else {
        board.player1_y()++;
        if (board.player0_y() == board.player1_y() &&
            board.player0_x() == board.player1_x()) {
          if (board.player1_y() < HEIGHT - 1 &&
              !board(1, board.player1_x(), board.player1_y())) {
            board.player1_y()++;
          } else if (board.player1_x() > 0 && board.player1_x() < WIDTH - 1 &&
                     !board(0, board.player1_x() - 1, board.player1_y()) &&
                     !board(0, board.player1_x(), board.player1_y())) {
            int dis_0 =
                shortestDistance(board.player1_x() - 1, board.player1_y(), 0);
            int dis_1 =
                shortestDistance(board.player1_x() + 1, board.player1_y(), 0);
            if (dis_0 <= dis_1) {
              board.player1_x()--;
            } else {
              board.player1_x()++;
            }
          } else if (board.player1_x() > 0 &&
                     !board(0, board.player1_x() - 1, board.player1_y())) {
            board.player1_x()--;
          } else if (board.player1_x() < WIDTH - 1 &&
                     !board(0, board.player1_x(), board.player1_y())) {
            board.player1_x()++;
          } else {
            board.player1_y()--;
          }
        }
      }
    } else if (action == MOVE_LEFT) {
      if (current_player == 0) {
        board.player0_x()--;
        if (board.player0_y() == board.player1_y() &&
            board.player0_x() == board.player1_x()) {
          if (board.player0_x() > 0 &&
              !board(0, board.player0_x() - 1, board.player0_y())) {
            board.player0_x()--;
          } else if (board.player0_y() > 0 && board.player0_y() < HEIGHT - 1 &&
                     !board(1, board.player0_x(), board.player0_y()) &&
                     !board(1, board.player0_x(), board.player0_y() - 1)) {
            int dis_0 = shortestDistance(board.player0_x(),
                                         board.player0_y() - 1, WIDTH - 1);
            int dis_1 = shortestDistance(board.player0_x(),
                                         board.player0_y() + 1, WIDTH - 1);
            if (dis_0 <= dis_1) {
              board.player0_y()--;
            } else {
              board.player0_y()++;
            }
          } else if (board.player0_y() < HEIGHT - 1 &&
                     !board(1, board.player0_x(), board.player0_y())) {
            board.player0_y()++;
          } else if (board.player0_y() > 0 &&
                     !board(1, board.player0_x(), board.player0_y() - 1)) {
            board.player0_y()--;
          } else {
            board.player0_x()++;
          }
        }
      } else {
        board.player1_x()++;
        if (board.player0_y() == board.player1_y() &&
            board.player0_x() == board.player1_x()) {
          if (board.player1_x() < WIDTH - 1 &&
              !board(0, board.player1_x(), board.player1_y())) {
            board.player1_x()++;
          } else if (board.player1_y() > 0 && board.player1_y() < HEIGHT - 1 &&
                     !board(1, board.player1_x(), board.player1_y()) &&
                     !board(1, board.player1_x(), board.player1_y() - 1)) {
            int dis_0 =
                shortestDistance(board.player1_x(), board.player1_y() - 1, 0);
            int dis_1 =
                shortestDistance(board.player1_x(), board.player1_y() + 1, 0);
            if (dis_0 < dis_1) {
              board.player1_y()--;
            } else {
              board.player1_y()++;
            }
          } else if (board.player1_y() < HEIGHT - 1 &&
                     !board(1, board.player1_x(), board.player1_y())) {
            board.player1_y()++;
          } else if (board.player1_y() > 0 &&
                     !board(1, board.player1_x(), board.player1_y() - 1)) {
            board.player1_y()--;
          } else {
            board.player1_x()--;
          }
        }
      }
    } else if (action == MOVE_RIGHT) {
      if (current_player == 0) {
        board.player0_x()++;
        if (board.player0_y() == board.player1_y() &&
            board.player0_x() == board.player1_x()) {
          if (board.player0_x() < WIDTH - 1 &&
              !board(0, board.player0_x(), board.player0_y())) {
            board.player0_x()++;
          } else if (board.player0_y() > 0 && board.player0_y() < HEIGHT - 1 &&
                     !board(1, board.player0_x(), board.player0_y()) &&
                     !board(1, board.player0_x(), board.player0_y() - 1)) {
            int dis_0 = shortestDistance(board.player0_x(),
                                         board.player0_y() - 1, WIDTH - 1);
            int dis_1 = shortestDistance(board.player0_x(),
                                         board.player0_y() + 1, WIDTH - 1);
            if (dis_0 < dis_1) {
              board.player0_y()--;
            } else {
              board.player0_y()++;
            }
          } else if (board.player0_y() < HEIGHT - 1 &&
                     !board(1, board.player0_x(), board.player0_y())) {
            board.player0_y()++;
          } else if (board.player0_y() > 0 &&
                     !board(1, board.player0_x(), board.player0_y() - 1)) {
            board.player0_y()--;
          } else {
            board.player0_x()--;
          }
        }
      } else {
        board.player1_x()--;
        if (board.player0_y() == board.player1_y() &&
            board.player0_x() == board.player1_x()) {
          if (board.player1_x() > 0 &&
              !board(0, board.player1_x() - 1, board.player1_y())) {
            board.player1_x()--;
          } else if (board.player1_y() > 0 && board.player1_y() < HEIGHT - 1 &&
                     !board(1, board.player1_x(), board.player1_y()) &&
                     !board(1, board.player1_x(), board.player1_y() - 1)) {
            int dis_0 =
                shortestDistance(board.player1_x(), board.player1_y() - 1, 0);
            int dis_1 =
                shortestDistance(board.player1_x(), board.player1_y() + 1, 0);
            if (dis_0 <= dis_1) {
              board.player1_y()--;
            } else {
              board.player1_y()++;
            }
          } else if (board.player1_y() < HEIGHT - 1 &&
                     !board(1, board.player1_x(), board.player1_y())) {
            board.player1_y()++;
          } else if (board.player1_y() > 0 &&
                     !board(1, board.player1_x(), board.player1_y() - 1)) {
            board.player1_y()--;
          } else {
            board.player1_x()++;
          }
        }
      }
    } else {
      if (action < (WIDTH - 1) * (HEIGHT - 1) + 4) {
        int x = (action - 4) / (HEIGHT - 1), y = (action - 4) % (HEIGHT - 1);
        if (current_player) x = WIDTH - 2 - x;
        board(0, x, y) = 1;
        board(0, x, y + 1) = 1;
        if (y) valid_moves(0, x, y - 1) = 0;
        valid_moves(0, x, y) = 0;
        valid_moves(0, x, y + 1) = 0;
        valid_moves(1, x, y) = 0;
      } else {
        int x = (action - 4 - (WIDTH - 1) * (HEIGHT - 1)) / (HEIGHT - 1),
            y = (action - 4 - (WIDTH - 1) * (HEIGHT - 1)) % (HEIGHT - 1);
        if (current_player) x = WIDTH - 2 - x;
        board(1, x, y) = 1;
        board(1, x + 1, y) = 1;
        if (x) valid_moves(1, x - 1, y) = 0;
        valid_moves(1, x, y) = 0;
        valid_moves(1, x + 1, y) = 0;
        valid_moves(0, x, y) = 0;
      }
      board.player_cnt(current_player)--;
    }
    current_player = !current_player;
  }

  bool End() {
    return (board.player0_x() == WIDTH - 1 || board.player1_x() == 0);
  }

  bool Winner() {
    assert(End());

    // We assert the game has ended here, so just check if player 1 has
    // reached end or not.
    return board.player1_x() == 0;
  }

  uint8_t player0_x() const noexcept { return board.player0_x(); }
  void set_player0_x(uint8_t x) noexcept { board.player0_x() = x; }
  uint8_t player1_x() const noexcept { return board.player1_x(); }
  void set_player1_x(uint8_t x) noexcept { board.player1_x() = x; }
  uint8_t player0_y() const noexcept { return board.player0_y(); }
  void set_player0_y(uint8_t y) noexcept { board.player0_y() = y; }
  uint8_t player1_y() const noexcept { return board.player1_y(); }
  void set_player1_y(uint8_t y) noexcept { board.player1_y() = y; }

  /*
    This function checks if there exist a path for both player.

    Note:
    1. current implementation used a simple algorithm that performs a DFS
    search. a better way should be maintaining a cache for the distance
    (length of shortest path) of every point on the board.
  */
  bool checkPathExist() const noexcept {
    static std::bitset<WIDTH * HEIGHT> flag0 = 0, flag1 = 0;
    static std::once_flag initFlags;
    std::call_once(initFlags, [&]() {
      for (int j = 0; j < HEIGHT; j++) {
        flag0.set(pos(WIDTH - 1, j));
      }
      for (int j = 0; j < HEIGHT; j++) {
        flag1.set(pos(0, j));
      }
    });

    std::bitset<WIDTH * HEIGHT> tmp = 0;

    int x = board.player0_x();
    int y = board.player0_y();
    tmp.set(pos(x, y));
    do {
      bool success = false;
      for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
          if (tmp.test(pos(i, j))) {
            if (i > 0 && !board(0, i - 1, j) && !tmp.test(pos(i - 1, j))) {
              tmp.set(pos(i - 1, j));
              success = true;
            }
            if (i < WIDTH - 1 && !board(0, i, j) && !tmp.test(pos(i + 1, j))) {
              tmp.set(pos(i + 1, j));
              success = true;
            }
            if (j > 0 && !board(1, i, j - 1) && !tmp.test(pos(i, j - 1))) {
              tmp.set(pos(i, j - 1));
              success = true;
            }
            if (j < HEIGHT - 1 && !board(1, i, j) && !tmp.test(pos(i, j + 1))) {
              tmp.set(pos(i, j + 1));
              success = true;
            }
          }
        }
      }
      if (!success) {
        return false;
      }
      if ((tmp & flag0).any()) break;
    } while (true);

    tmp.reset();
    x = board.player1_x();
    y = board.player1_y();

    // the target of player 1 is to go to left of the board, with a size of
    // 9x9
    tmp.set(pos(x, y));
    do {
      bool success = false;
      for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
          if (tmp.test(pos(i, j))) {
            if (i > 0 && !board(0, i - 1, j) && !tmp.test(pos(i - 1, j))) {
              tmp.set(pos(i - 1, j));
              success = true;
            }
            if (i < WIDTH - 1 && !board(0, i, j) && !tmp.test(pos(i + 1, j))) {
              tmp.set(pos(i + 1, j));
              success = true;
            }
            if (j > 0 && !board(1, i, j - 1) && !tmp.test(pos(i, j - 1))) {
              tmp.set(pos(i, j - 1));
              success = true;
            }
            if (j < HEIGHT - 1 && !board(1, i, j) && !tmp.test(pos(i, j + 1))) {
              tmp.set(pos(i, j + 1));
              success = true;
            }
          }
        }
      }
      if (!success) return false;
      if ((tmp & flag1).any()) break;
    } while (true);

    return true;
  }

  /*
  This function calculates the shortest distance to get to column [col] from
  (x, y).
*/
  int shortestDistance(int x, int y, int col) const {
    if (x == col) return 0;
    if (!(x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT))
      return HEIGHT * WIDTH + 1;  // slightly handle error case
    std::queue<std::pair<int, int>> q;
    bool vis[HEIGHT * WIDTH] = {0};
    q.push(std::make_pair(x, y));
    vis[x * WIDTH + y] = 1;
    int cnt_x = 0;
    q.push({-1, -1});
    while (!q.empty()) {
      auto [x, y] = q.front();
      q.pop();
      if (x == -1) {
        cnt_x++;
        if (!q.empty()) q.push({-1, -1});
        continue;
      }
      if (x == col) {
        break;
      }
      if (x < WIDTH - 1 && !board(0, x, y) && !vis[(x + 1) * HEIGHT + y]) {
        q.push({x + 1, y});
        vis[(x + 1) * HEIGHT + y] = 1;
      }
      if (x > 0 && !board(0, x - 1, y) && !vis[(x - 1) * HEIGHT + y]) {
        q.push({x - 1, y});
        vis[(x - 1) * HEIGHT + y] = 1;
      }
      if (y < HEIGHT - 1 && !board(1, x, y) && !vis[x * HEIGHT + y + 1]) {
        q.push({x, y + 1});
        vis[x * HEIGHT + y + 1] = 1;
      }
      if (y > 0 && !board(1, x, y - 1) && !vis[x * HEIGHT + y - 1]) {
        q.push({x, y - 1});
        vis[x * HEIGHT + y - 1] = 1;
      }
    }
    return cnt_x;
  }

  void Canonicalize(float *storage) const {
    /* Note that currently we create storage from torch::from_zeros(), so memset
    is not required here. */
    // memset(storage, 0, sizeof(float) * CANONICAL_SHAPE[0] *
    // CANONICAL_SHAPE[1] * CANONICAL_SHAPE[2]);
    float(&out)[CANONICAL_SHAPE[0]][CANONICAL_SHAPE[1]][CANONICAL_SHAPE[2]] =
        *reinterpret_cast<float(*)[CANONICAL_SHAPE[0]][CANONICAL_SHAPE[1]]
                                  [CANONICAL_SHAPE[2]]>(storage);

    for (int w = 0; w < WIDTH - 1; w++) {
      for (int h = 0; h < HEIGHT; h++) {
        out[0][w][h] = board(0, w, h);
      }
    }
    for (int w = 0; w < WIDTH; w++) {
      for (int h = 0; h < HEIGHT - 1; h++) {
        out[1][w][h] = board(1, w, h);
      }
    }
    out[2][board.player0_x()][board.player0_y()] = 10;
    out[3][board.player1_x()][board.player1_y()] = 10;

    // TODO: may optimize here.
    auto valids = Valid_moves();
    for (auto i = 0; i < WIDTH - 1; ++i) {
      for (auto j = 0; j < HEIGHT - 1; ++j) {
        if (valids[4 + i * (HEIGHT - 1) + j]) {
          out[4][current_player ? WIDTH - 2 - i : i][j] = 1;
        }
        if (valids[4 + (WIDTH - 1) * (HEIGHT - 1) + i * (HEIGHT - 1) + j]) {
          out[5][current_player ? WIDTH - 2 - i : i][j] = 1;
        }
      }
    }

    std::fill(&out[6][0][0], &out[7][0][0], current_player);

    for (int i = 0; i < 9; i++) {
      out[7][0][i] = board.player0_cnt() > i;
      out[7][1][i] = board.player1_cnt() > i;
    }
    int cnt_now = shortestDistance(board.player0_x(), board.player0_y(), 8);
    out[7][2][0] = cnt_now / 16.;
    if (current_player == 1 || valids[MOVE_UP]) {
      int upx, upy, cnt_up = -1;
      upx = board.player0_x();
      upy = board.player0_y() - 1;
      if (upx == board.player1_x() && upy == board.player1_y()) {
        if (upy > 0 && !board(1, upx, upy - 1)) {
          upy--;
        } else if (upx > 0 && upx < 8 && !board(0, upx, upy) &&
                   !board(0, upx - 1, upy)) {
          int dis_0 = shortestDistance(upx - 1, upy, WIDTH - 1);
          int dis_1 = shortestDistance(upx + 1, upy, WIDTH - 1);
          if (dis_0 < dis_1) {
            upx--;
            cnt_up = dis_0;
          } else {
            upx++;
            cnt_up = dis_1;
          }
        } else if (upx < 8 && !board(0, upx, upy)) {
          upx++;
        } else if (upx > 0 && !board(0, upx - 1, upy)) {
          upx--;
        } else {
          upy++;
        }
      }
      if (cnt_up == -1) cnt_up = shortestDistance(upx, upy, 8);
      if (cnt_up < cnt_now) {
        out[7][2][1] = 1;
      }
    }
    if (current_player == 1 || valids[MOVE_DOWN]) {
      int downx, downy, cnt_down = -1;
      downx = board.player0_x();
      downy = board.player0_y() + 1;
      if (downy == board.player1_y() && downx == board.player1_x()) {
        if (downy < 8 && !board(1, downx, downy)) {
          downy++;
        } else if (downx > 0 && downx < 8 && !board(0, downx, downy) &&
                   !board(0, downx - 1, downy)) {
          int dis_0 = shortestDistance(downx - 1, downy, WIDTH - 1);
          int dis_1 = shortestDistance(downx + 1, downy, WIDTH - 1);
          if (dis_0 < dis_1) {
            downx--;
            cnt_down = dis_0;
          } else {
            downx++;
            cnt_down = dis_1;
          }
        } else if (downx < 8 && !board(0, downx, downy)) {
          downx++;
        } else if (downx > 0 && !board(0, downx - 1, downy)) {
          downx--;
        } else {
          downy--;
        }
      }
      if (cnt_down == -1) cnt_down = shortestDistance(downx, downy, 8);
      if (cnt_down < cnt_now) {
        out[7][2][2] = 1;
      }
    }
    if (current_player == 1 || valids[MOVE_LEFT]) {
      int leftx, lefty, cnt_left = -1;
      leftx = board.player0_x() - 1;
      lefty = board.player0_y();
      if (lefty == board.player1_y() && leftx == board.player1_x()) {
        if (leftx > 0 && !board(0, leftx - 1, lefty)) {
          leftx--;
        } else if (lefty > 0 && lefty < 8 && !board(1, leftx, lefty) &&
                   !board(1, leftx, lefty - 1)) {
          int dis_0 = shortestDistance(leftx, lefty - 1, WIDTH - 1);
          int dis_1 = shortestDistance(leftx, lefty + 1, WIDTH - 1);
          if (dis_0 <= dis_1) {
            lefty--;
            cnt_left = dis_0;
          } else {
            lefty++;
            cnt_left = dis_1;
          }
        } else if (lefty < 8 && !board(1, leftx, lefty)) {
          lefty++;
        } else if (lefty > 0 && !board(1, leftx, lefty - 1)) {
          lefty--;
        } else {
          leftx++;
        }
      }
      if (cnt_left == -1) cnt_left = shortestDistance(leftx, lefty, 8);
      if (cnt_left < cnt_now) {
        out[7][2][3] = 1;
      }
    }
    if (current_player == 1 || valids[MOVE_RIGHT]) {
      int rightx, righty, cnt_right = -1;
      rightx = board.player0_x() + 1;
      righty = board.player0_y();
      if (righty == board.player1_y() && rightx == board.player1_x()) {
        if (rightx < 8 && !board(0, rightx, righty)) {
          rightx++;
        } else if (righty > 0 && righty < 8 && !board(1, rightx, righty) &&
                   !board(1, rightx, righty - 1)) {
          int dis_0 = shortestDistance(rightx, righty - 1, WIDTH - 1);
          int dis_1 = shortestDistance(rightx, righty + 1, WIDTH - 1);
          if (dis_0 < dis_1) {
            righty--;
            cnt_right = dis_0;
          } else {
            righty++;
            cnt_right = dis_1;
          }
        } else if (righty < 8 && !board(1, rightx, righty)) {
          righty++;
        } else if (righty > 0 && !board(1, rightx, righty - 1)) {
          righty--;
        } else {
          rightx--;
        }
      }
      if (cnt_right == -1) cnt_right = shortestDistance(rightx, righty, 8);
      if (cnt_right < cnt_now) {
        out[7][2][4] = 1;
      }
    }

    cnt_now = shortestDistance(board.player1_x(), board.player1_y(), 0);
    out[7][3][0] = cnt_now / 16.;
    if (current_player == 0 || valids[MOVE_UP]) {
      int upx, upy, cnt_up = -1;
      upx = board.player1_x();
      upy = board.player1_y() - 1;
      if (board.player0_y() == upy && board.player0_x() == upx) {
        if (upy > 0 && !board(1, upx, upy - 1)) {
          upy--;
        } else if (upx > 0 && upx < 8 && !board(0, upx - 1, upy) &&
                   !board(0, upx, upy)) {
          int dis_0 = shortestDistance(upx - 1, upy, 0);
          int dis_1 = shortestDistance(upx + 1, upy, 0);
          if (dis_0 <= dis_1) {
            upx--;
            cnt_up = dis_0;
          } else {
            upx++;
            cnt_up = dis_1;
          }
        } else if (upx > 0 && !board(0, upx - 1, upy)) {
          upx--;
        } else if (upx < 8 && !board(0, upx, upy)) {
          upx++;
        } else {
          upy++;
        }
      }
      if (cnt_up == -1) cnt_up = shortestDistance(upx, upy, 0);
      if (cnt_up < cnt_now) {
        out[7][2][5] = 1;
      }
    }
    if (current_player == 0 || valids[MOVE_DOWN]) {
      int downx, downy, cnt_down = -1;
      downx = board.player1_x();
      downy = board.player1_y() + 1;
      if (board.player0_y() == downy && board.player0_x() == downx) {
        if (downy < 8 && !board(1, downx, downy)) {
          downy++;
        } else if (downx > 0 && downx < 8 && !board(0, downx - 1, downy) &&
                   !board(0, downx, downy)) {
          int dis_0 = shortestDistance(downx - 1, downy, 0);
          int dis_1 = shortestDistance(downx + 1, downy, 0);
          if (dis_0 <= dis_1) {
            downx--;
            cnt_down = dis_0;
          } else {
            downx++;
            cnt_down = dis_1;
          }
        } else if (downx > 0 && !board(0, downx - 1, downy)) {
          downx--;
        } else if (downx < 8 && !board(0, downx, downy)) {
          downx++;
        } else {
          downy--;
        }
      }
      if (cnt_down == -1) cnt_down = shortestDistance(downx, downy, 0);
      if (cnt_down < cnt_now) {
        out[7][2][6] = 1;
      }
    }
    if (current_player == 0 || valids[MOVE_LEFT]) {
      int leftx, lefty, cnt_left = -1;
      leftx = board.player1_x() + 1;
      lefty = board.player1_y();
      if (board.player0_y() == lefty && board.player0_x() == leftx) {
        if (leftx < 8 && !board(0, leftx, lefty)) {
          leftx++;
        } else if (lefty > 0 && lefty < 8 && !board(1, leftx, lefty) &&
                   !board(1, leftx, lefty - 1)) {
          int dis_0 = shortestDistance(leftx, lefty - 1, 0);
          int dis_1 = shortestDistance(leftx, lefty + 1, 0);
          if (dis_0 < dis_1) {
            lefty--;
            cnt_left = dis_0;
          } else {
            lefty++;
            cnt_left = dis_1;
          }
        } else if (lefty < 8 && !board(1, leftx, lefty)) {
          lefty++;
        } else if (lefty > 0 && !board(1, leftx, lefty - 1)) {
          lefty--;
        } else {
          leftx--;
        }
      }
      if (cnt_left == -1) cnt_left = shortestDistance(leftx, lefty, 0);
      if (cnt_left < cnt_now) {
        out[7][2][7] = 1;
      }
    }
    if (current_player == 0 || valids[MOVE_RIGHT]) {
      int rightx, righty, cnt_right = -1;
      rightx = board.player1_x() - 1;
      righty = board.player1_y();
      if (board.player0_y() == righty && board.player0_x() == rightx) {
        if (rightx > 0 && !board(0, rightx - 1, righty)) {
          rightx--;
        } else if (righty > 0 && righty < 8 && !board(1, rightx, righty) &&
                   !board(1, rightx, righty - 1)) {
          int dis_0 = shortestDistance(rightx, righty - 1, 0);
          int dis_1 = shortestDistance(rightx, righty + 1, 0);
          if (dis_0 <= dis_1) {
            righty--;
            cnt_right = dis_0;
          } else {
            righty++;
            cnt_right = dis_1;
          }
        } else if (righty < 8 && !board(1, rightx, righty)) {
          righty++;
        } else if (righty > 0 && !board(1, rightx, righty - 1)) {
          righty--;
        } else {
          rightx++;
        }
      }
      if (cnt_right == -1) cnt_right = shortestDistance(rightx, righty, 0);
      if (cnt_right < cnt_now) {
        out[7][2][8] = 1;
      }
    }
  }

  std::string ToString() const noexcept {
    std::string out = current_player ? "Current Player: X" : "Current Player: O";
    out += "\n    1   2   3   4   5   6   7   8   9 \n  +---+---+---+---+---+---+---+---+---+\n";
    for (int i = 0; i < 9; i++) {
      out += (char)('1' + i);
      out += " |";
      for (int j = 0; j < 9; j++) {
        if (board.player0_x() == j && board.player0_y() == i) {
          out += " O ";
        } else if (board.player1_x() == j && board.player1_y() == i) {
          out += " X ";
        } else {
          out += "   ";
        }
        if (j == 8 || board(0, j, i)) {
          out += '|';
        } else {
          out += ' ';
        }
      }
      out += '\n';
      if (i < 8) {
        out += "  |";
        for (int j = 0; j < 9; j++) {
          if (board(1, j, i)) {
            out += "---";
          } else {
            out += "   ";
          }
          if (j < 8) {
            out += '+';
          }
        }
        out += "+ ";
        out += (char)('i' + i);
        out += '\n';
      }
    }
    out += "  +---+---+---+---+---+---+---+---+---+\n      a   b   c   d   e   f   g   h\n";
    out += std::format("walls left = {}, {}\n", board.player0_cnt(), board.player1_cnt());
    return out;
  }

  static std::string action_to_string(ActionType action, bool player) {
    return Quoridor::action_to_string(action, player);
  }

  static ActionType string_to_action(const std::string &str, bool player) {
    return Quoridor::string_to_action(str, player);
  }
};
}  // namespace Quoridor

#include <iostream>

#include "core/examples/game_quoridor.h"
#include "gtest/gtest.h"
#include "torch/torch.h"

using namespace Quoridor;

/* Test case #1:
 * Perform simple checks of the beginning board:
 * 1. player position
 * 2. path exist
 * 3. valid moves
 */
TEST(QuoridorTest, SimpleTest) {
  GameState game;
  EXPECT_EQ(game.player0_x(), 0);
  EXPECT_EQ(game.player0_y(), PLAYER_0_INIT_Y);
  EXPECT_EQ(game.player1_x(), WIDTH - 1);
  EXPECT_EQ(game.player1_y(), PLAYER_1_INIT_Y);
  EXPECT_TRUE(game.checkPathExist());
  auto valid_moves = game.Valid_moves();
  int cnt = 0;
  for (auto i : valid_moves) cnt += i;
  EXPECT_EQ(cnt, 3 + (HEIGHT - 1) * (WIDTH - 1) * 2);
  EXPECT_FALSE(valid_moves[MOVE_LEFT]);
  EXPECT_TRUE(valid_moves[MOVE_RIGHT]);
}

/* Test case #2:
 * Perform simple checks of the beginning board, but player 1 goes first:
 * 1. player position
 * 2. path exist
 * 3. valid moves
 */
TEST(QuoridorTest, SimpleTest2) {
  GameState game;
  game.Move(MOVE_PASS);

  EXPECT_EQ(game.player0_x(), 0);
  EXPECT_EQ(game.player0_y(), PLAYER_0_INIT_Y);
  EXPECT_EQ(game.player1_x(), WIDTH - 1);
  EXPECT_EQ(game.player1_y(), PLAYER_1_INIT_Y);
  EXPECT_TRUE(game.checkPathExist());
  auto valid_moves = game.Valid_moves();
  int cnt = 0;
  for (auto i : valid_moves) cnt += i;
  EXPECT_EQ(cnt, 3 + (HEIGHT - 1) * (WIDTH - 1) * 2);
  EXPECT_FALSE(valid_moves[MOVE_LEFT]);
  EXPECT_TRUE(valid_moves[MOVE_RIGHT]);
}

/* Test case #3:
 * This test calculates Hash() result of the beginning board and all
 * situations after one move, without MOVE_LEFT, but include MOVE_PASS.
 * These hash values should all be different.
 */
TEST(QuoridorTest, HashTest) {
  GameState game;
  std::set<uint64_t> hashes;
  hashes.insert(game.Hash());
  for (ActionType i = 0; i < NUM_ACTIONS; i++) {
    if (i == MOVE_LEFT) continue;
    auto game_ = game;
    game_.Move(i);
    hashes.insert(game_.Hash());
  }
  game.Move(MOVE_PASS);
  hashes.insert(game.Hash());
  EXPECT_EQ(hashes.size(), NUM_ACTIONS + 1);
}

/* Test case #4:
 * This test calculates Hash() result of the beginning board and all
 * following situations when every player rush to the end.
 * These hash values should all be different.
 */
TEST(QuoridorTest, HashTest2) {
  GameState game;
  std::set<uint64_t> hashes;
  hashes.insert(game.Hash());
  while (!game.End()) {
    game.Move(MOVE_RIGHT);
    hashes.insert(game.Hash());
  }
  unsigned expected_size = WIDTH * 2 - (WIDTH % 2 == 0 ? 4 : 3);
  EXPECT_EQ(hashes.size(), expected_size);
}

/* Test case #5:
 * This tests valid moves of three situations:
 * 1. Initial board
 * 2. After putting one horizontal wall.
 * 3. After putting another horizontal wall.
 */
TEST(QuoridorTest, ValidMoves) {
  if (WIDTH < 7 || HEIGHT < 5) {
    GTEST_SKIP();
  }

  GameState game;
  auto valids = game.Valid_moves();
  std::vector<uint8_t> expected(NUM_ACTIONS, 0);
  expected[MOVE_RIGHT] = expected[MOVE_UP] = expected[MOVE_DOWN] = 1;
  for (int i = 4; i < NUM_ACTIONS; i++) expected[i] = 1;
  EXPECT_EQ(valids, expected);

  game.Move(h_wall(2, 2));

  expected[h_wall_op(1, 2)] = 0;
  expected[h_wall_op(2, 2)] = 0;
  expected[h_wall_op(3, 2)] = 0;
  expected[v_wall_op(2, 2)] = 0;
  valids = game.Valid_moves();
  EXPECT_EQ(valids, expected);
  expected[h_wall_op(1, 2)] = 1;
  expected[h_wall_op(2, 2)] = 1;
  expected[h_wall_op(3, 2)] = 1;
  expected[v_wall_op(2, 2)] = 1;

  game.Move(h_wall_op(4, 2));

  expected[h_wall(1, 2)] = 0;
  expected[h_wall(2, 2)] = 0;
  expected[h_wall(3, 2)] = 0;
  expected[h_wall(4, 2)] = 0;
  expected[h_wall(5, 2)] = 0;
  expected[v_wall(2, 2)] = 0;
  expected[v_wall(4, 2)] = 0;
  valids = game.Valid_moves();
  EXPECT_EQ(valids, expected);
}

/* Test case #6:
 * This tests valid moves of three situations:
 * 1. Initial board
 * 2. After putting one vertical wall.
 * 3. After putting another vertical wall.
 */
TEST(QuoridorTest, ValidMoves2) {
  if (WIDTH < 5 || HEIGHT < 7) {
    GTEST_SKIP();
  }

  GameState game;
  auto valids = game.Valid_moves();
  std::vector<uint8_t> expected(NUM_ACTIONS, 0);
  expected[MOVE_RIGHT] = expected[MOVE_UP] = expected[MOVE_DOWN] = 1;
  for (int i = 4; i < NUM_ACTIONS; i++) expected[i] = 1;
  EXPECT_EQ(valids, expected);

  game.Move(v_wall(2, 2));

  expected[v_wall_op(2, 1)] = 0;
  expected[v_wall_op(2, 2)] = 0;
  expected[v_wall_op(2, 3)] = 0;
  expected[h_wall_op(2, 2)] = 0;
  valids = game.Valid_moves();
  EXPECT_EQ(valids, expected);
  expected[v_wall_op(2, 1)] = 1;
  expected[v_wall_op(2, 2)] = 1;
  expected[v_wall_op(2, 3)] = 1;
  expected[h_wall_op(2, 2)] = 1;

  game.Move(v_wall_op(2, 4));

  expected[v_wall(2, 1)] = 0;
  expected[v_wall(2, 2)] = 0;
  expected[v_wall(2, 3)] = 0;
  expected[v_wall(2, 4)] = 0;
  expected[v_wall(2, 5)] = 0;
  expected[h_wall(2, 2)] = 0;
  expected[h_wall(2, 4)] = 0;
  valids = game.Valid_moves();
  EXPECT_EQ(valids, expected);
}

/* Test case #7:
 * This tests the Valid_moves() and Move() when performing a jumping down.
 */
TEST(QuoridorTest, JumpTest) {
  if (WIDTH < 5) {
    GTEST_SKIP();
  }

  GameState game;
  game.set_player0_x(3);
  game.set_player0_y(0);
  game.set_player1_x(3);
  game.set_player1_y(1);
  game.Move(v_wall(2, 0));
  game.Move(MOVE_PASS);

  // Valid path of player 0 doesn't contain up and left, because he is at a
  // corner.
  {
    auto valids = game.Valid_moves();
    EXPECT_EQ(valids[MOVE_UP], 0);
    EXPECT_EQ(valids[MOVE_DOWN], 1);
    EXPECT_EQ(valids[MOVE_LEFT], 0);
    EXPECT_EQ(valids[MOVE_RIGHT], 1);
  }

  // Perform a jumpover.
  game.Move(MOVE_DOWN);

  EXPECT_EQ(game.player0_x(), 3);
  EXPECT_EQ(game.player0_y(), 2);

  // Valid path of player 1 doesn't contain right, because he is beside a wall.
  {
    auto valids = game.Valid_moves();
    EXPECT_EQ(valids[MOVE_UP], 1);
    EXPECT_EQ(valids[MOVE_DOWN], 1);
    EXPECT_EQ(valids[MOVE_LEFT], 1);
    EXPECT_EQ(valids[MOVE_RIGHT], 0);
  }
}

/* Test case #8:
 * This tests the Valid_moves() and Move() when performing a jumping up.
 */
TEST(QuoridorTest, JumpTest2) {
  if (WIDTH < 5 || HEIGHT != 9) {
    GTEST_SKIP();
  }

  GameState game;
  game.set_player0_x(3);
  game.set_player0_y(8);
  game.set_player1_x(3);
  game.set_player1_y(7);
  game.Move(v_wall(2, 7));
  game.Move(MOVE_PASS);

  // Valid path of player 0 doesn't contain down and left, because he is at a
  // corner.
  {
    auto valids = game.Valid_moves();
    EXPECT_EQ(valids[MOVE_UP], 1);
    EXPECT_EQ(valids[MOVE_DOWN], 0);
    EXPECT_EQ(valids[MOVE_LEFT], 0);
    EXPECT_EQ(valids[MOVE_RIGHT], 1);
  }

  // Perform a jumpover.
  game.Move(MOVE_UP);

  EXPECT_EQ(game.player0_x(), 3);
  EXPECT_EQ(game.player0_y(), 6);

  // Valid path of player 1 doesn't contain right, because he is beside a wall.
  {
    auto valids = game.Valid_moves();
    EXPECT_EQ(valids[MOVE_UP], 1);
    EXPECT_EQ(valids[MOVE_DOWN], 1);
    EXPECT_EQ(valids[MOVE_LEFT], 1);
    EXPECT_EQ(valids[MOVE_RIGHT], 0);
  }
}

/* Test case #9:
 * This tests the Valid_moves() when player can jump to left/right.
 */
TEST(QuoridorTest, JumpTest3) {
  if (WIDTH != 9 || HEIGHT < 5) {
    GTEST_SKIP();
  }

  GameState game;
  game.set_player0_x(7);
  game.set_player0_y(3);
  game.set_player1_x(8);
  game.set_player1_y(3);
  game.Move(v_wall(7, 3));
  game.Move(MOVE_PASS);

  {
    auto valids = game.Valid_moves();
    EXPECT_EQ(valids[MOVE_UP], 1);
    EXPECT_EQ(valids[MOVE_DOWN], 1);
    EXPECT_EQ(valids[MOVE_LEFT], 1);
    EXPECT_EQ(valids[MOVE_RIGHT], 0);
  }
  game.Move(MOVE_PASS);

  {
    auto valids = game.Valid_moves();
    EXPECT_EQ(valids[MOVE_UP], 1);
    EXPECT_EQ(valids[MOVE_DOWN], 1);
    EXPECT_EQ(valids[MOVE_LEFT], 0);
    EXPECT_EQ(valids[MOVE_RIGHT], 0);
  }
}

/* Test case #10:
 * This tests the Valid_moves() when player can jump to left/right.
 */
TEST(QuoridorTest, JumpTest4) {
  if (WIDTH < 5 || HEIGHT < 5) {
    GTEST_SKIP();
  }

  GameState game;
  game.set_player0_x(0);
  game.set_player0_y(3);
  game.set_player1_x(1);
  game.set_player1_y(3);
  game.Move(h_wall(0, 2));
  game.Move(MOVE_PASS);

  {
    auto valids = game.Valid_moves();
    EXPECT_EQ(valids[MOVE_UP], 0);
    EXPECT_EQ(valids[MOVE_DOWN], 1);
    EXPECT_EQ(valids[MOVE_LEFT], 0);
    EXPECT_EQ(valids[MOVE_RIGHT], 1);
  }

  game.Move(MOVE_PASS);
  {
    auto valids = game.Valid_moves();
    EXPECT_EQ(valids[MOVE_UP], 0);
    EXPECT_EQ(valids[MOVE_DOWN], 1);
    EXPECT_EQ(valids[MOVE_LEFT], 1);
    EXPECT_EQ(valids[MOVE_RIGHT], 1);
  }
}

/* Test case #11:
 * This tests the Move() when player perform a jump in front of a wall. The
 * player should jump along the shortest path.
 * This is a jump-up test.
 */
TEST(QuoridorTest, JumpTest5) {
  if (WIDTH < 5) {
    GTEST_SKIP();
  }

  GameState game;
  game.set_player0_x(3);
  game.set_player0_y(0);
  game.set_player1_x(3);
  game.set_player1_y(1);
  game.Move(v_wall(2, 0));
  game.Move(MOVE_UP);

  EXPECT_EQ(game.player1_x(), 4);
  EXPECT_EQ(game.player1_y(), 0);
}

/* Test case #12:
 * This tests the Move() when player perform a jump in front of a wall. The
 * player should jump along the shortest path.
 * This is a jump-down test.
 */
TEST(QuoridorTest, JumpTest6) {
  if (WIDTH < 4) {
    GTEST_SKIP();
  }

  GameState game;
  game.set_player0_x(1);
  game.set_player0_y(0);
  game.set_player1_x(1);
  game.set_player1_y(1);
  game.Move(h_wall(1, 1));
  game.Move(MOVE_PASS);

  game.Move(MOVE_DOWN);
  EXPECT_EQ(game.player0_x(), 2);
  EXPECT_EQ(game.player0_y(), 1);
}

/* Test case #13:
 * This is a jump-down test similiar to #12, but adding a wall should make the
 * player jump to another side.
 */
TEST(QuoridorTest, JumpTest7) {
  if (WIDTH < 4) {
    GTEST_SKIP();
  }

  GameState game;
  game.set_player0_x(1);
  game.set_player0_y(0);
  game.set_player1_x(1);
  game.set_player1_y(1);
  game.Move(h_wall(1, 1));
  game.Move(v_wall_op(2, 0));

  game.Move(MOVE_DOWN);
  EXPECT_EQ(game.player0_x(), 0);
  EXPECT_EQ(game.player0_y(), 1);
}

/* Test case #14:
 * This tests the rule that both player should have a valid path.
 * Player 0 stays in the left bottom corner, and some walls are not valid
 * because they block his way out.
 */
TEST(QuoridorTest, ValidMoves3) {
  if (HEIGHT != 9) {
    GTEST_SKIP();
  }

  GameState game;
  game.set_player0_x(0);
  game.set_player0_y(8);
  game.Move(v_wall(0, 5));
  game.Move(h_wall_op(0, 4));
  game.Move(v_wall(1, 7));

  auto valids = game.Valid_moves();
  EXPECT_FALSE(valids[h_wall_op(0, 6)]);
  EXPECT_FALSE(valids[h_wall_op(0, 7)]);
  EXPECT_FALSE(valids[h_wall_op(1, 6)]);
}

/* Test case #15:
 * This tests the rule that both player should have a valid path.
 * Player 0 stays in the left top corner, and some walls are not valid
 * because they block his way out.
 */
TEST(QuoridorTest, ValidMoves4) {
  if (WIDTH < 4 || HEIGHT < 4) {
    GTEST_SKIP();
  }

  GameState game;
  game.set_player0_x(0);
  game.set_player0_y(0);
  game.Move(h_wall(0, 1));
  game.Move(h_wall_op(2, 0));
  game.Move(v_wall(3, 0));

  auto valids = game.Valid_moves();
  EXPECT_FALSE(valids[v_wall_op(0, 0)]);
  EXPECT_FALSE(valids[v_wall_op(0, 1)]);
  EXPECT_TRUE(valids[v_wall_op(0, 2)]);

  EXPECT_FALSE(valids[v_wall_op(1, 0)]);
  EXPECT_FALSE(valids[v_wall_op(1, 1)]);
  EXPECT_TRUE(valids[v_wall_op(1, 2)]);

  EXPECT_FALSE(valids[h_wall_op(0, 0)]);
  EXPECT_FALSE(valids[h_wall_op(1, 0)]);
  EXPECT_FALSE(valids[h_wall_op(2, 0)]);
  EXPECT_FALSE(valids[h_wall_op(2, 1)]);
  EXPECT_TRUE(valids[h_wall_op(2, 2)]);
}

/* Test case #16:
 * This tests the rule that both player should have a valid path.
 * Player 1 stays in the right top corner, and some walls are not valid
 * because they block his way out.
 */
TEST(QuoridorTest, ValidMoves5) {
  if (WIDTH != 9) {
    GTEST_SKIP();
  }

  GameState game;
  game.set_player1_x(8);
  game.set_player1_y(1);
  game.Move(h_wall(3, 1));
  game.Move(h_wall_op(5, 1));
  game.Move(h_wall(7, 1));

  auto valids = game.Valid_moves();
  EXPECT_TRUE(valids[v_wall_op(0, 0)]);
  EXPECT_TRUE(valids[v_wall_op(1, 0)]);
  EXPECT_FALSE(valids[v_wall_op(2, 0)]);
  EXPECT_FALSE(valids[v_wall_op(3, 0)]);
  EXPECT_FALSE(valids[v_wall_op(4, 0)]);
  EXPECT_FALSE(valids[v_wall_op(5, 0)]);
  EXPECT_FALSE(valids[v_wall_op(6, 0)]);
  EXPECT_FALSE(valids[v_wall_op(7, 0)]);
}

/* Test case #17:
 * This tests the copy constructor function.
 * Copied game should have same hash as the original game.
 */
TEST(QuoridorTest, CopyTest) {
  std::vector<GameState> games;
  GameState game;
  games.push_back(game);
  for (int i = 0; i < 10; i++) {
    // pick a random valid action
    auto valids = game.Valid_moves();
    ActionType action = rand() % NUM_ACTIONS;
    while (!valids[action]) action = rand() % NUM_ACTIONS;
    game.Move(action);
    for (auto& g : games) g.Move(action);
    games.push_back(game);
  }
  for (auto& g : games) {
    EXPECT_EQ(g.Hash(), game.Hash());
  }
}

/* Test case #18:
 * This tests the Copy() function.
 * Copied game should have same hash as the original game.
 */
TEST(QuoridorTest, CopyTest2) {
  std::vector<std::unique_ptr<GameState>> games;
  GameState game;
  games.push_back(game.Copy());
  for (int i = 0; i < 10; i++) {
    // pick a random valid action
    auto valids = game.Valid_moves();
    ActionType action = rand() % NUM_ACTIONS;
    while (!valids[action]) action = rand() % NUM_ACTIONS;
    game.Move(action);
    for (auto& g : games) g->Move(action);
    games.push_back(game.Copy());
  }
  for (auto& g : games) {
    EXPECT_EQ(g->Hash(), game.Hash());
  }
}

/* Test case #19:
 * This tests the Canonicalize() function.
 */
TEST(QuoridorTest, CanonicalizeTest) {
  if (WIDTH != 9 || HEIGHT != 9) {
    GTEST_SKIP();
  }

  GameState game;
  for(int i = 0; i < 20; i++) {
    auto valid_moves = game.Valid_moves();
    int j = i * 20;
    while(!valid_moves[j % NUM_ACTIONS]) j++;
    game.Move(j % NUM_ACTIONS);
  }
  auto canonicalized = torch::zeros({1, CANONICAL_SHAPE[0], CANONICAL_SHAPE[1],
                                     CANONICAL_SHAPE[2]});
  game.Canonicalize(canonicalized.data_ptr<float>());

  std::ifstream in("testdata/quoridor_canonical_testcase.pt", std::ios::binary);
  EXPECT_TRUE(in.is_open()) << "Cannot open testdata/quoridor_canonical_testcase.pt";
  std::vector<char> buffer(std::istreambuf_iterator<char>(in), {});

  auto expected = torch::pickle_load(buffer).toTensor();

  EXPECT_TRUE(torch::allclose(canonicalized, expected));
}

/* Test case #99:
 * This function tests the functionality of action_to_string() and
 * string_to_action().
 * Note that these two functions do not check for input validity.
 */
TEST(QuoridorTest, Act2Str) {
  EXPECT_EQ(action_to_string(4), "a12");
  for (ActionType i = 0; i < NUM_ACTIONS; i++) {
    EXPECT_EQ(i, string_to_action(action_to_string(i)));
  }
}

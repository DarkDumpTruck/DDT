#include <gtest/gtest.h>

#include "core/util/common.h"
#include "core/util/floating.h"
#include "core/util/lemke_howson.h"

TEST(LemkeHowsonTest, SimpleTest) {
  const int dim1 = 3;
  const int dim2 = 2;
  auto bimatrix = create_bimatrix(dim1, dim2);
  bimatrix[0][0] = -1;
  bimatrix[0][1] = 2;
  bimatrix[1][0] = 1;
  bimatrix[1][1] = -1;
  bimatrix[2][0] = -3;
  bimatrix[2][1] = 6;
  bimatrix[3][0] = 1;
  bimatrix[3][1] = -2;
  bimatrix[4][0] = -1;
  bimatrix[4][1] = 1;
  bimatrix[5][0] = 3;
  bimatrix[5][1] = -6;
  auto result = solve_equilibrium(bimatrix, dim1, dim2);
  // the equilibrium of above unique: (0, 9/11, 2/11, 7/11, 4/11)
  EXPECT_DOUBLE_EQ(result[0], 0);
  EXPECT_DOUBLE_EQ(result[1], 9.0 / 11);
  EXPECT_DOUBLE_EQ(result[2], 2.0 / 11);
  EXPECT_DOUBLE_EQ(result[3], 7.0 / 11);
  EXPECT_DOUBLE_EQ(result[4], 4.0 / 11);
}

TEST(LemkeHowsonTest, SimpleTestPositive) {
  const int dim1 = 3;
  const int dim2 = 2;
  auto bimatrix = create_bimatrix(dim1, dim2);
  bimatrix[0][0] = 9;
  bimatrix[0][1] = 12;
  bimatrix[1][0] = 11;
  bimatrix[1][1] = 9;
  bimatrix[2][0] = 7;
  bimatrix[2][1] = 16;
  bimatrix[3][0] = 11;
  bimatrix[3][1] = 8;
  bimatrix[4][0] = 9;
  bimatrix[4][1] = 11;
  bimatrix[5][0] = 13;
  bimatrix[5][1] = 4;
  auto result = solve_equilibrium_positive(bimatrix, dim1, dim2);
  // the equilibrium of above unique: (0, 9/11, 2/11, 7/11, 4/11)
  EXPECT_DOUBLE_EQ(result[0], 0);
  EXPECT_DOUBLE_EQ(result[1], 9.0 / 11);
  EXPECT_DOUBLE_EQ(result[2], 2.0 / 11);
  EXPECT_DOUBLE_EQ(result[3], 7.0 / 11);
  EXPECT_DOUBLE_EQ(result[4], 4.0 / 11);
}

TEST(LemkeHowsonTest, SimpleTest2) {
  const int dim1 = 3;
  const int dim2 = 3;
  auto bimatrix = create_bimatrix(dim1, dim2);
  bimatrix[0][0] = 6;
  bimatrix[0][1] = 3;
  bimatrix[0][2] = 5;
  bimatrix[1][0] = 3;
  bimatrix[1][1] = 1;
  bimatrix[1][2] = 7;
  bimatrix[2][0] = 4;
  bimatrix[2][1] = 6;
  bimatrix[2][2] = 6;
  for (int i = 3; i < 6; i++) {
    for (int j = 0; j < 3; j++) {
      bimatrix[i][j] = -bimatrix[i - 3][j];
    }
  }
  auto result = solve_equilibrium(bimatrix, dim1, dim2);
  std::cout << "Result of SimpleTest2: ";
  for (int i = 0; i < 6; i++) {
    std::cout << result[i] << " ";
  }
  std::cout << std::endl;
  // the equilibrium of above unique: (0.4, 0, 0.6, 0.6, 0.4, 0)
  EXPECT_DOUBLE_EQ(result[0], 0.4);
  EXPECT_DOUBLE_EQ(result[1], 0);
  EXPECT_DOUBLE_EQ(result[2], 0.6);
  EXPECT_DOUBLE_EQ(result[3], 0.6);
  EXPECT_DOUBLE_EQ(result[4], 0.4);
  EXPECT_DOUBLE_EQ(result[5], 0);
}

TEST(LemkeHowsonTest, SimpleTest3) {
  const int dim1 = 3;
  const int dim2 = 3;
  auto bimatrix = create_bimatrix(dim1, dim2);
  bimatrix[0][0] = 0;
  bimatrix[0][1] = 2;
  bimatrix[0][2] = 0;
  bimatrix[1][0] = 2;
  bimatrix[1][1] = 0;
  bimatrix[1][2] = 2;
  bimatrix[2][0] = 1;
  bimatrix[2][1] = 1;
  bimatrix[2][2] = 0;
  bimatrix[3][0] = 2;
  bimatrix[3][1] = 0;
  bimatrix[3][2] = 0;
  bimatrix[4][0] = 0;
  bimatrix[4][1] = 2;
  bimatrix[4][2] = 0;
  bimatrix[5][0] = 0;
  bimatrix[5][1] = 1;
  bimatrix[5][2] = 1;
  auto result = solve_equilibrium(bimatrix, dim1, dim2);
  // this is not a zero-sum game, and equilibrium is not unique
  // there are two equilibria: (1/3, 0, 2/3, 1/2, 1/2, 0) or (1/2, 1/2, 0, 1/2,
  // 1/2, 0)
  EXPECT_TRUE(fabs(result[0] - 1.0 / 3) < Eps ||
              fabs(result[0] - 1.0 / 2) < Eps);
  EXPECT_TRUE(fabs(result[1] - 0) < Eps || fabs(result[1] - 1.0 / 2) < Eps);
  EXPECT_TRUE(fabs(result[2] - 2.0 / 3) < Eps || fabs(result[2] - 0) < Eps);
  EXPECT_DOUBLE_EQ(result[3], 0.5);
  EXPECT_DOUBLE_EQ(result[4], 0.5);
  EXPECT_DOUBLE_EQ(result[5], 0);
}

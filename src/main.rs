use anyhow::Context;
use itertools::Itertools;
use rand::seq::SliceRandom;
use std::collections::HashMap;

// Some nice helper functions

fn coords(index: i8) -> (i8, i8) {
    (index % 4, index / 4)
}

fn taxicab_norm(a: (i8, i8), b: (i8, i8)) -> i8 {
    (a.0 - b.0).abs() + (a.1 - b.1).abs()
}

fn distance(a: i8, b: i8) -> i8 {
    let a = coords(a);
    let b = coords(b);
    taxicab_norm(a, b)
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
enum Move {
    Up,
    Down,
    Left,
    Right,
}

impl Move {
    fn from_char(c: char) -> Option<Move> {
        match c {
            'w' => Some(Move::Up),
            's' => Some(Move::Down),
            'a' => Some(Move::Left),
            'd' => Some(Move::Right),
            _ => None,
        }
    }

    fn index_update(&self, index: i8) -> Option<i8> {
        let (x, y) = coords(index);
        match self {
            Move::Up => {
                if y == 0 {
                    None
                } else {
                    Some((y - 1) * 4 + x)
                }
            }
            Move::Down => {
                if y == 3 {
                    None
                } else {
                    Some((y + 1) * 4 + x)
                }
            }
            Move::Left => {
                if x == 0 {
                    None
                } else {
                    Some(y * 4 + (x - 1))
                }
            }
            Move::Right => {
                if x == 3 {
                    None
                } else {
                    Some(y * 4 + (x + 1))
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Board {
    order: [i8; 16],
}

impl Board {
    fn new() -> Board {
        // permute 0..=15 (15 is the blank)
        let order = std::array::from_fn::<i8, 16, _>(|i| i as i8);
        let mut board = Board { order };

        board.shuffle(1000000); // This should be good enough, right??
        board
    }

    fn piece_location(&self, piece: i8) -> i8 {
        self.order.iter().position(|p| *p == piece).unwrap() as i8
    }

    fn valid_moves(&self) -> Vec<Move> {
        use Move::*;
        let mut moves = Vec::new();

        let blank_position = self.piece_location(15);
        let (x, y) = coords(blank_position);

        if x != 0 {
            moves.push(Left);
        }
        if x != 3 {
            moves.push(Right);
        }

        if y != 0 {
            moves.push(Up);
        }
        if y != 3 {
            moves.push(Down);
        }

        moves
    }

    /// Less is better
    fn score(&self) -> i8 {
        self.order
            .iter()
            .take(15) // don't score the blank
            .enumerate()
            .map(|(i, location)| {
                let piece = *location;
                let correct = i as i8;
                distance(piece, correct)
            })
            .sum()
    }

    fn correct(&self) -> i8 {
        self.order
            .iter()
            .enumerate()
            .filter(|(i, &piece)| *i as i8 == piece)
            .count() as i8
    }

    fn solved(&self) -> bool {
        self.correct() == 16
    }

    // count how many contiguous pieces are in the correct position from 0
    fn solved_in_order_from_zero(&self) -> usize {
        let mut correct = 0;
        for (i, &piece) in self.order.iter().enumerate() {
            if i as i8 == piece {
                correct += 1;
            } else {
                break;
            }
        }
        correct
    }

    #[allow(dead_code)]
    fn print(&self) {
        for y in 0..4 {
            for x in 0..4 {
                let index = y * 4 + x;
                let piece = self.order[index];
                if piece == 15 {
                    print!("   ");
                } else {
                    print!("{:2} ", piece + 1);
                }
            }
            println!();
        }
    }

    fn execute_move(&mut self, r#move: Move) {
        let blank = self.order.iter().position(|p| *p == 15).unwrap() as i8;

        if !self.valid_moves().contains(&r#move) {
            return;
        }

        if let Some(new_blank) = Move::index_update(&r#move, blank) {
            self.order.swap(new_blank as usize, blank as usize);
        } else {
            panic!("Invalid move");
        }
    }

    fn shuffle(&mut self, n: usize) {
        let mut rng = rand::thread_rng();
        for _ in 0..n {
            let valid_moves = self.valid_moves();
            let r#move = valid_moves.choose(&mut rng).unwrap();
            self.execute_move(r#move.clone());
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
struct SubProblem {
    restricted: usize, // encoding pieces 0..restricted
    goal: usize,       // encoding pieces restricted..(restricted + goal)
}

impl SubProblem {
    fn relax(&mut self) {
        self.restricted = self
            .restricted
            .checked_sub(1)
            .expect("already maximally unrestricted.");
        self.goal += 1;
    }
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
enum Piece {
    Restricted(i8),
    Goal(i8),
    Blank,
    Ignored,
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
struct SubState {
    board: Vec<Piece>, // length 16 None for don't cares or already solved
}

impl SubState {
    fn blank_index(&self) -> i8 {
        self.board
            .iter()
            .position(|p| matches!(p, Piece::Blank))
            .unwrap() as i8
    }

    fn next_state(&self, action: Move) -> Self {
        let blank_index = self.blank_index();
        let new_blank_index = action.index_update(blank_index).unwrap();

        let mut piece_locations = self.board.clone();
        piece_locations.swap(blank_index as usize, new_blank_index as usize);

        Self {
            board: piece_locations,
        }
    }

    fn valid_moves(&self) -> Vec<Move> {
        use Move::*;
        [Up, Down, Left, Right]
            .into_iter()
            .filter(|r#move| {
                let new_blank = r#move.index_update(self.blank_index());
                new_blank.is_some_and(|new_blank| {
                    // check if the new blank does not displace a restricted piece
                    !matches!(self.board[new_blank as usize], Piece::Restricted(_))
                })
            })
            .collect()
    }
    fn from_board(subproblem: SubProblem, board: &Board) -> Self {
        let r = subproblem.restricted;
        let g = subproblem.goal;

        Self {
            board: board
                .order
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    if i < r {
                        assert!(i as i8 == p);
                        Piece::Restricted(p)
                    } else if (p as usize) < (r + g) {
                        Piece::Goal(p)
                    } else if p == 15 {
                        Piece::Blank
                    } else {
                        Piece::Ignored
                    }
                })
                .collect(),
        }
    }

    fn solved(&self) -> bool {
        self.board.iter().enumerate().all(|(i, p)| match p {
            Piece::Goal(n) | Piece::Restricted(n) => i as i8 == *n,
            _ => true,
        })
    }
}

type ValueFunction = HashMap<SubState, f64>;

impl SubProblem {
    fn gen_states(&self) -> ValueFunction {
        // produce every distinguishable ordering of the restricted pieces
        let mut states = ValueFunction::new();

        for mut permutation in (self.restricted..16)
            .permutations(self.goal + 1) // distribute the goal pieces and the blank
            .map(Vec::into_iter)
        {
            let blank_index = permutation.next().unwrap();
            let goal_indicies = permutation.collect::<Vec<_>>();
            let pieces = (self.restricted..(self.restricted + self.goal))
                .map(|n| Piece::Goal(n as i8))
                .collect::<Vec<_>>();

            // enter restricted pieces
            let mut piece_locations = (0..self.restricted)
                .map(|n| Piece::Restricted(n as i8))
                .collect::<Vec<_>>();

            // fill rest with ignored
            piece_locations.resize(16, Piece::Ignored);

            for (index, piece) in goal_indicies.iter().zip(pieces) {
                piece_locations[*index] = piece;
            }

            // assign blank
            piece_locations[blank_index] = Piece::Blank;

            states.insert(
                SubState {
                    board: piece_locations,
                },
                0.0,
            );
        }
        states
    }
    fn reward(&self, substate: SubState) -> f64 {
        for (index, piece) in substate.board.iter().enumerate() {
            match piece {
                Piece::Goal(n) => {
                    if index as i8 != *n {
                        return 0.0;
                    }
                }
                Piece::Restricted(n) => {
                    if index as i8 != *n {
                        unreachable!("Restricted piece not in correct location");
                    }
                }
                _ => {}
            }
        }
        1.0
    }

    fn value_iteration(self: SubProblem) -> ValueFunction {
        let gamma = 0.8; // TODO
        let mut value_function = self.gen_states();

        loop {
            let mut cumulative_value_delta = 0.0;

            let value_function_last = value_function.clone();
            for (state, value) in value_function.iter_mut() {
                let moves = state.valid_moves();

                let moves_and_values = moves.into_iter().map(|m| {
                    let newstate = state.next_state(m.clone());
                    (
                        m,
                        value_function_last
                            .get(&newstate)
                            .with_context(|| format!("State not in value function: {:?}", newstate))
                            .unwrap(),
                    )
                });

                // TODO check
                let best_move_value = moves_and_values
                    .fold((None, f64::MIN), |(best_move, best_value), (m, v)| {
                        if *v > best_value {
                            (Some(m), *v)
                        } else {
                            (best_move, best_value)
                        }
                    })
                    .1;

                let new_value = self.reward(state.clone()) + gamma * best_move_value;
                cumulative_value_delta += new_value - *value;
                *value = new_value;
            }

            //println!("Cumulative value delta: {}", cumulative_value_delta);
            let threshold = 0.0001; // TODO set threshold
            if cumulative_value_delta < threshold {
                // TODO set threshold
                //println!("Converged to within cumulative value delta {}", threshold);
                break;
            }
        }
        value_function
    }
}

fn policy(value_function: &ValueFunction, state: &SubState) -> Move {
    let moves = state.valid_moves();

    let moves_and_values = moves.into_iter().map(|m| {
        let newstate = state.next_state(m.clone());
        (
            m,
            value_function
                .get(&newstate)
                .expect("State not in value function"),
        )
    });

    // TODO check
    moves_and_values
        .fold((None, f64::MIN), |(best_move, best_value), (m, v)| {
            if *v > best_value {
                (Some(m), *v)
            } else {
                (best_move, best_value)
            }
        })
        .0
        .unwrap()
}

fn heuristic(board: &mut Board) -> usize {
    // get input char, convert to move, execute move, print board

    let mut seen = std::collections::HashSet::<Board>::new();

    let mut steps = 0;

    loop {
        if board.solved() {
            break;
        }
        steps += 1;

        let valid_moves = board.valid_moves();

        if let Some((best_move, _score, new_board)) = valid_moves
            .iter()
            .filter_map(|r#move| {
                let mut new_board = *board; // .clone()??
                new_board.execute_move(r#move.clone());
                if seen.contains(&new_board) {
                    None
                } else {
                    Some((r#move, new_board.score(), new_board))
                }
            })
            .min_by_key(|(_, score, _)| *score)
        {
            board.execute_move(best_move.clone());
            seen.insert(new_board);
        } else {
            // pick random
            let random_move = valid_moves.choose(&mut rand::thread_rng()).unwrap();
            board.execute_move(random_move.clone());
        }
    }
    steps
}

fn value_iteration(board: &mut Board) -> usize {
    let min_goal = 2;
    let mut steps = 0;

    let mut r = 0;
    while r < 15 {
        r = board.solved_in_order_from_zero();
        let g = usize::min(min_goal, 15 - r);
        let mut subproblem = SubProblem {
            restricted: r,
            goal: g,
        };

        let mut value_function = subproblem.value_iteration();
        let mut substate = SubState::from_board(subproblem, board);

        while *value_function
            .get(&substate)
            .with_context(|| {
                format!(
                    "State not in value function: {:?}... value function: {:?}",
                    substate,
                    value_function.keys().count()
                )
            })
            .unwrap()
            <= 0.01
        // should be good enough to mitigate floating point errors
        {
            subproblem.relax();
            substate = SubState::from_board(subproblem, board);
            value_function = subproblem.value_iteration();
        }

        //board.print();
        loop {
            if substate.solved() {
                //println!("Solved subproblem {:?}", subproblem);
                break;
            }

            let action = policy(&value_function, &substate);
            steps += 1;
            //println!("{}: {:?}", steps, action);
            board.execute_move(action.clone());
            substate = substate.next_state(action);
        }
        r += 1;

        if board.solved() {
            break;
        }
    }
    //board.print();

    //println!("Solved in {} steps", steps);
    steps
}

fn main() {
    let n = 10000;
    println!("Creating {} boards", n);
    let mut boards1 = vec![Board::new(); n];
    let mut boards2 = boards1.clone();
    println!("Created {} boards", n);

    // time
    let begin = std::time::Instant::now();
    let mut steps1 = 0;
    for board in &mut boards1 {
        steps1 += heuristic(board);
    }
    let end = std::time::Instant::now();
    let time1 = end - begin;

    let begin = std::time::Instant::now();
    let mut steps2 = 0;
    for board in &mut boards2 {
        steps2 += value_iteration(board);
    }
    let end = std::time::Instant::now();
    let time2 = end - begin;

    println!("Heuristic: {} steps in {:?}", steps1 / n, time1 / n as u32);
    println!(
        "Value iteration: {} steps in {:?}",
        steps2 / n,
        time2 / n as u32
    );
}

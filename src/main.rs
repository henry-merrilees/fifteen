use rand::seq::SliceRandom;

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

#[derive(PartialEq, Debug, Copy, Clone)]
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Board {
    order: [i8; 16],
}

impl Board {
    fn new() -> Board {
        // permute 0..=15 (15 is the blank)
        let mut rng = rand::thread_rng();
        let mut order = std::array::from_fn::<i8, 16, _>(|i| i as i8);
        order.shuffle(&mut rng);

        let blank_row = order.iter().position(|&p| p == 15).unwrap() as i8 / 4;
        let blank_row_parity = (blank_row + 1) % 2;

        let mut inversions = 0;

        for i in 0..15 {
            for j in (i + 1)..16 {
                let first = order[i];
                let second = order[j];
                if (first > second) && (first != 15 && second != 15) {
                    inversions += 1;
                }
            }
        }
        let inversion_parity = inversions % 2;

        if inversion_parity ^ blank_row_parity == 1 {
            // find two non-blank pieces and swap them
            println!("Swapping");
            let mut i = 0;
            let mut j = 1;
            while order[i] == 15 || order[j] == 15 {
                i += 1;
                j += 1;
            }
            println!("Swapping {} and {}", i, j);
        }

        Board { order }
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

    fn score_diff(&self, r#move: Move) -> i8 {
        // assumes move in bounds
        let blank = self.order[15];

        let new_blank = blank
            - match r#move {
                Move::Up => 4,
                Move::Down => -4,
                Move::Left => 1,
                Move::Right => -1,
            } % 16;

        let piece = self.order.iter().position(|p| *p == new_blank).unwrap() as i8;

        distance(blank, piece) - distance(new_blank, piece)
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

        let new_blank = blank
            - match r#move {
                Move::Up => 4,
                Move::Down => -4, // -4 % 16
                Move::Left => 1,
                Move::Right => -1, // -1 % 16
            } % 16;

        self.order.swap(new_blank as usize, blank as usize);
    }
}

fn main() {
    // get input char, convert to move, execute move, print board

    let mut board = Board::new();

    let term = console::Term::stdout();

    let mut seen = std::collections::HashSet::<Board>::new();

    let mut i = 0;

    loop {
        i += 1;
        // board.print();
        // term.clear_last_lines(4).unwrap();

        if board.solved() {
            println!("Solved in {} moves", i);
            break;
        }

        let valid_moves = board.valid_moves();

        if let Some((best_move, _score, new_board)) = valid_moves
            .iter()
            .filter_map(|r#move| {
                let mut new_board = board.clone();
                new_board.execute_move(*r#move);
                if seen.contains(&new_board) {
                    None
                } else {
                    Some((r#move, new_board.score(), new_board))
                }
            })
            .min_by_key(|(_, score, _)| *score)
        {
            // println!("Move: {:?}", best_move);
            board.execute_move(*best_move);
            seen.insert(new_board);
        } else {
            // pick random
            let random_move = valid_moves.choose(&mut rand::thread_rng()).unwrap();
            // println!("Rand: {:?}", random_move);
            board.execute_move(*random_move);
        }
    }
}

import unittest
from critic_selection import learned_critic_rows


def record(group, terminal, target):
    return dict(group_index=group,target=target,metadata=dict(lane='critic',
        terminal_boundary=terminal,diagnostics=dict(mean_return=target)))


class SelectionTests(unittest.TestCase):
    def test_known_boundaries_do_not_drown_successful_root(self):
        root = record(0,False,1.)
        zero_nonterminal = record(1,False,0.)
        records = [root,*[record(0,True,0.) for _ in range(128)],zero_nonterminal]
        learned,report = learned_critic_rows(records)
        self.assertEqual(learned,[root,zero_nonterminal])
        self.assertIs(learned[0],root)
        self.assertEqual(report['groups']['0'],dict(exported=129,learned=1,fixed_terminal=128))

    def test_unknown_or_nonzero_terminal_boundary_rejected(self):
        for row in (record(0,None,0.),record(0,True,1.)):
            with self.assertRaises(ValueError):
                learned_critic_rows([record(0,False,.5),row])

    def test_every_question_retains_a_learned_target(self):
        with self.assertRaises(ValueError):
            learned_critic_rows([record(0,False,.5),record(1,True,0.)])


if __name__ == '__main__':
    unittest.main()

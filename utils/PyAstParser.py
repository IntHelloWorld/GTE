import os
import re
from functools import lru_cache
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple

from tqdm import tqdm
from tree_sitter import Language, Node, Parser, Tree


class PyAstParser:
    def __init__(self, so, language):
        """Init PyAstParser and build parser.

        Args:
            so (str): The .so file generated for building parser. See https://github.com/tree-sitter/py-tree-sitter.

            language (str): The target language of the parser.
        """
        self.parser = Parser()
        LANGUAGE = Language(so, language)
        self.parser.set_language(LANGUAGE)

    def file2ast(self, file: str, print_error=False) -> Tree:
        """Parse single file to single ast.
        Args:
            file (str): the absolute path of parsed file.

        Return:
            ast (tree_sitter.Tree): the ast of parsed file.

        """
        path = Path(file)
        assert path.is_file()
        with path.open(encoding="utf-8") as f:
            text = f.read()
            ast = self.parser.parse(bytes(text, "utf-8"))
            if ast.root_node.has_error:
                if print_error:
                    print(f"Warning! {str(path)} parsing error")
        return ast

    def jsonl2ast(self, file: str) -> Dict[str, Dict[str, Tree]]:
        """
        Parse jsonl file to asts dict.
        :param file: the absolute path of jsonl file.
        :return: asts with form Dict[dir_name, Dict[id, Tree]]
        """
        import json

        asts = {}
        dir_name = str(Path(file).parent)
        with open(file, "r") as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                func = js["func"]
                idx = js["idx"]
                ast = self.parser.parse(bytes(func, "utf-8"))
                if ast.root_node.has_error:
                    # try with wrapped temp class
                    func = "public class TempClass {" + func + "}"
                    ast = self.parser.parse(bytes(func, "utf-8"))
                    if ast.root_node.has_error:
                        print(f"Warning: idx{idx} parsing error!")
                try:
                    asts[dir_name][idx] = ast
                except Exception:
                    asts[dir_name] = {idx: ast}
        return asts

    def dir2asts(self, dir: str, tranverse: bool = False) -> Dict[str, Dict[str, Tree]]:
        """Parse all files in the dir to asts dict.

        Args:
            dir (str): the path of the dir contains files to be parsed.
            tranverse (bool): if tranverse all the files in dir.

        Return:
            asts (Dict[str, Dict[str, Tree]]): asts with form Dict[dir_name, Dict[file_name, Tree]].
        """
        asts = dict()

        def all_path(path):
            for sub_path in path.iterdir():
                if sub_path.is_file():
                    yield sub_path
                else:
                    for sub_sub_path in all_path(sub_path):
                        yield sub_sub_path
            return "done"

        if tranverse:
            for path in tqdm(all_path(Path(dir)), desc="itering dirs and parsing files..."):
                ast = self.file2ast(str(path))
                file_name = str(path.name)
                dir_name = str(path.parent)
                try:
                    asts[dir_name][file_name] = ast
                except Exception:
                    asts[dir_name] = {file_name: ast}
        else:
            for path in tqdm(Path(dir).iterdir(), desc="parsing files..."):
                if path.is_file():
                    ast = self.file2ast(str(path))
                    file_name = str(path.name)
                    dir_name = str(path.parent)
                    try:
                        asts[dir_name][file_name] = ast
                    except Exception:
                        asts[dir_name] = {file_name: ast}
        return asts

    @staticmethod
    def is_ignored_node(node: Node):
        """Return True if the input node need to be ignored."""
        if "comment" in node.type:
            return True
        return False

    @staticmethod
    def asts2sequences(asts: Dict[str, Dict[str, Tree]]) -> Dict[str, List[str]]:
        """Turning parsed asts to token sequences, also known as 'flattered', can be used for training word2vec.

        Args:
            asts (Dict[str, Dict[str, Tree]]): Parsed asts.

        Returns:
            Dict[str, List[str]]: Sequences with form Dict[file_path, List[tokens]].
        """

        def get_sequence(node, sequence):
            token = PyAstParser.get_token(node)
            children = PyAstParser.get_children(node)
            if token:
                sequence.append(token)
            for child in children:
                get_sequence(child, sequence)

        sequences = {}
        for dir_name in tqdm(asts, desc="asts to sequences"):
            for fname, ast in asts[dir_name].items():
                seq = []
                get_sequence(ast.root_node, seq)
                fpath = os.path.join(dir_name, fname)
                sequences[fpath] = seq

        return sequences

    @staticmethod
    @lru_cache(maxsize=5000)
    def split_identifier(identifier: str) -> List[str]:
        """Split a single identifier into parts on snake_case and camelCase.

        Args:
            identifier (str): The identifier.

        Returns:
            List[str]: The sub-tokens.
        """
        REGEX_TEXT = (
            "(?<=[a-z0-9])(?=[A-Z])|"
            "(?<=[A-Z0-9])(?=[A-Z][a-z])|"
            "(?<=[0-9])(?=[a-zA-Z])|"
            "(?<=[A-Za-z])(?=[0-9])|"
            "(?<=[@$.'\"])(?=[a-zA-Z0-9])|"
            "(?<=[a-zA-Z0-9])(?=[@$.'\"])|"
            "_|\\s+"
        )
        import sys

        if sys.version_info >= (3, 7):
            import re

            SPLIT_REGEX = re.compile(REGEX_TEXT)
        else:
            import regex

            SPLIT_REGEX = regex.compile("(?V1)" + REGEX_TEXT)
        identifier_parts = list(s.lower() for s in SPLIT_REGEX.split(identifier) if len(s) > 0)

        if len(identifier_parts) == 0:
            return [identifier]
        return identifier_parts

    @staticmethod
    def get_token(node: Node, lower: bool = False):
        """Get the token of an ast node, the token of a leaf node is its text in code, the token of a non-leaf node is its ast type.
        If subtoken==True, return the list of subtokens instead. Return None if fail to get the token.

        Args:
            node (tree_sitter.Node): Ast node.
            lower (bool): If True, the tokens will be lowered. Defaults to False.

        Returns:
            token (str|None): The token of the input node. Return None otherwise.
        """
        assert not PyAstParser.is_ignored_node(node)
        if len(PyAstParser.get_children(node)) == 0:  # leaf node
            if "literal" in node.type:
                token = node.type
            else:
                token = re.sub(r"\s", "", str(node.text, "utf-8"))
                if token == "":
                    token = node.type
        else:
            token = node.type
        assert not token == ""
        if lower:
            token = token.lower()
        return token

    @staticmethod
    def get_subtokens(node: Node):
        """Get the list of subtokens. Return None if fail to get the token.

        Args:
            node (tree_sitter.Node): Ast node.

        Returns:
            subtokens (List[str]|None): The token of the input node. Return None otherwise.
        """
        if PyAstParser.is_ignored_node(node):
            return None
        else:
            if len(PyAstParser.get_children(node)) == 0:  # leaf node
                if "literal" in node.type:
                    token = [node.type.lower()]
                else:
                    token = PyAstParser.split_identifier(str(node.text, "utf-8"))
            else:
                token = [node.type.lower()]
        if len(token) == 1 and token[0] == "":
            return None
        return token

    @staticmethod
    def get_child_with_type(node: Node, type: str, vague=False) -> Tuple[Node, int]:
        """Get the child and its index in all of the children with given type. Notice that the comment children are ignored.

        Args:
            node (tree_sitter.Node): Ast node.
            type (str): Expect type name. The pattern if vague is True.
            vague (bool): If vague mode. Default to False.

        Returns:
            child (tree_sitter.Node): Child of the node with given type. Return None if did not find.
            id (int): Child index. Return None if did not find.
        """
        id = 0
        if vague:
            for child in PyAstParser.get_children(node):
                if re.search(type, child.type) is not None:
                    return child, id
                id += 1
        else:
            for child in PyAstParser.get_children(node):
                if child.type == type:
                    return child, id
                id += 1
        return None, None

    @staticmethod
    def get_children(node: Node) -> List[Node]:
        """Get children of a ast node that not been ignored. Ignored nodes defined in PyAstParser.is_ignored_node().

        Args:
            node (tree_sitter.Node): Ast node.

        Returns:
            children (List[tree_sitter.Node]): Named children list of the input node.
        """
        children = []
        for child in node.children:
            if not PyAstParser.is_ignored_node(child):
                children.append(child)
        return children

    @staticmethod
    def distinguish_for(node: Node) -> str:
        """
        As for_statement has different types base on the presence of init, condition and update statements, this function distinguish the type of a for_statement.

        Args:
            node (tree_sitter.Node): input node, must be for_statement.

        Returns:
            type (str): type of the for_statement node, one of {"","i","ic","iu","cu","icu"}, i, c, u are abbrevations of init, condition and update respectively.
        """
        assert node.type == "for_statement"
        res = ""
        for child in PyAstParser.get_children(node):
            if child.type == "(":
                if child.next_sibling.type != ";":
                    res += "i"
                if child.next_sibling.next_sibling.type != ";":
                    res += "c"
            elif child.type == ")":
                if child.prev_sibling.type != ";":
                    res += "u"
        return res

    @staticmethod
    def distinguish_if(node: Node) -> str:
        """
        As if_statement has different types base on the presence of 'else if' and 'else', this function distinguish the type of a if_statement.

        Args:
            node (tree_sitter.Node): input node, must be if_statement.

        Returns:
            type (str): type of the for_statement node, one of {"if","if_elif","if_else"}.
        """
        assert node.type == "if_statement"
        for child in PyAstParser.get_children(node):
            if child.type == "else":
                if child.next_sibling.type == "if_statement":
                    return "if_elif"
                else:
                    return "if_else"
        return "if"

    @staticmethod
    def asts2token_vocab(
        asts: Dict[str, Dict[str, Tree]], pad_token: str = "<PAD>", subtoken: bool = False, statastic: bool = False
    ) -> Dict[str, int]:
        """Transform asts dict to a ast token vocabulary, ignore comments.

        Args:
            asts (Dict[str, Dict[str, Tree]]): dict with form Dict[dir_name, Dict[file_name, Tree]].
            pad_token (str): Pad token. Defaults to '<PAD>'.
            subtoken (bool): If True, the tokens will be splited into subtokens by snake_case and camelCase. Defaults to False.
            statastic (bool): If print the statastic information. Defaults to False.

        Return:
            token_vocab (Dict[str, int]): dict where keys are ast tokens, values are ids.
        """

        def get_subtoken_sequence(node, sequence):
            subtokens = PyAstParser.get_subtokens(node)
            children = PyAstParser.get_children(node)
            if subtokens:
                sequence.extend(subtokens)
            for child in children:
                get_subtoken_sequence(child, sequence)

        def get_token_sequence(node, sequence):
            token = PyAstParser.get_token(node)
            children = PyAstParser.get_children(node)
            if token:
                sequence.append(token)
            for child in children:
                get_token_sequence(child, sequence)

        def token_statistic(all_tokens):
            count = dict()
            for token in all_tokens:
                try:
                    count[token] += 1
                except Exception:
                    count[token] = 1
            return count

        all_tokens = []
        for dir_name in tqdm(asts, desc="Get token sequence"):
            for file_name, ast in asts[dir_name].items():
                if subtoken:
                    get_subtoken_sequence(ast.root_node, all_tokens)
                else:
                    get_token_sequence(ast.root_node, all_tokens)

        # Token statastic
        if statastic:
            count = token_statistic(all_tokens)
            print(f"Tokens quantity: {len(all_tokens)}")
            pprint(count)

        tokens = list(set(all_tokens))
        tokens.insert(0, pad_token)
        vocabsize = len(tokens)
        tokenids = range(vocabsize)
        token_vocab = dict(zip(tokens, tokenids))
        return token_vocab

    @staticmethod
    def asts2type_vocab(asts: Dict[str, Dict[str, Tree]], empty_type: str = "<EMPTY>") -> Dict[str, int]:
        """Transform asts dict to a ast types vocabulary.

        Args:
            asts (Dict[str, Dict[str, Tree]]): dict with form Dict[dir_name, Dict[file_name, Tree]].
            empty_type (str): Pad type. Defaults to '<EMPTY>'.

        Return:
            type_vocab (Dict[str, int]): dict where keys are ast types, values are ids.
        """

        def get_type_sequence(node, sequence):
            type = node.type
            children = PyAstParser.get_children(node)
            sequence.append(type)
            for child in children:
                get_type_sequence(child, sequence)

        all_types = []
        for dir_name in tqdm(asts, desc="Get token sequence"):
            for file_name, ast in asts[dir_name].items():
                get_type_sequence(ast.root_node, all_types)

        types = list(set(all_types))
        types.insert(0, empty_type)
        vocabsize = len(types)
        tokenids = range(vocabsize)
        type_vocab = dict(zip(types, tokenids))
        return type_vocab

    @staticmethod
    def ast2any_tree(tree: Tree, subtoken: bool = False):
        """Turn ast to anytree.  Require package 'anytree'.

        Args:
            tree (tree_sitter.Tree): The root node of the giving ast tree.
            subtoken (bool): If True, tokens will be splited into list of subtokens.

        Returns:
            newtree (AnyNode): The root node of the generated anytree.
        """
        from anytree import AnyNode

        global id
        id = 0

        def create_tree(node, parent):
            if subtoken:
                token = PyAstParser.get_subtokens(node)
            else:
                token = PyAstParser.get_token(node)
            global id
            if id > 0:
                newnode = AnyNode(id=id, token=token, type=node.type, data=node, parent=parent)
            else:
                newnode = parent
            id += 1
            children = PyAstParser.get_children(node)
            for child in children:
                create_tree(child, parent=newnode)

        root_node = tree.root_node
        if subtoken:
            root_token = PyAstParser.get_subtokens(root_node)
        else:
            root_token = PyAstParser.get_token(root_node)
        new_tree = AnyNode(id=id, token=root_token, type=root_node.type, data=root_node)
        create_tree(root_node, new_tree)
        return new_tree

    @staticmethod
    def ast2any_tree_pt(tree: Tree):
        from anytree import AnyNode

        global id
        id = 0
        global not_named_id
        not_named_id = 0

        def create_tree(node, parent):
            children = [c for c in node.children if "comment" not in c.type]
            token = PyAstParser.get_token(node)
            global id
            if id > 0:
                if node.is_named:
                    newnode = AnyNode(id=id, token=token, data=node, parent=parent)
                    id += 1
                else:
                    global not_named_id
                    newnode = AnyNode(id=f"nn{not_named_id}", token=token, data=node, parent=parent)
                    not_named_id += 1
            else:
                newnode = parent
                id += 1
            for child in children:
                create_tree(child, parent=newnode)

        root_node = tree.root_node
        root_token = PyAstParser.get_token(root_node)
        new_tree = AnyNode(id=id, token=root_token, data=root_node)
        create_tree(root_node, new_tree)
        return new_tree

    @staticmethod
    def tree_for_point(ast, p_batch: int = 1):
        from anytree import AnyNode

        def cal_tree_size(tree):
            return len([d for d in tree.descendants if d.token is not None]) + 1

        def visit_tree(samples, node):
            if len(samples) < p_batch:
                if isinstance(node.id, int):
                    if node.id % step == 0:
                        id, infos = point_sampler(node)
                        samples[id] = infos
            for child in node.children:
                visit_tree(samples, child)

        def to_class(n, type):
            # dist = {
            #     "size": [0, 165, 205, 250, 310, 450],
            #     "leaves": [0, 110, 135, 165, 205, 300],
            #     "not_leaves": [0, 56, 68, 82, 100, 150],
            # }
            dist = {
                "size": [0, 205, 310],
                "leaves": [0, 135, 205],
                "not_leaves": [0, 68, 100],
            }
            for idx, num in enumerate(dist[type]):
                if idx == len(dist[type]) - 1:
                    return idx
                if num < n <= dist[type][idx + 1]:
                    return idx

        def point_sampler(node: AnyNode):
            height = node.height
            leaves = len(node.leaves)
            size = len(node.descendants) + 1
            not_leaves = size - leaves
            return node.id, (height, to_class(size, "size"), to_class(leaves, "leaves"), to_class(not_leaves, "not_leaves"))

        new_tree = PyAstParser.ast2any_tree_pt(ast)
        ast_size = cal_tree_size(new_tree)
        assert p_batch <= ast_size
        step = ast_size // p_batch  # the step for point task sampler

        point_samples = {}
        visit_tree(point_samples, new_tree)
        assert len(point_samples) == p_batch
        return new_tree, point_samples

    @staticmethod
    def trees2DGLgraphs(asts, token_vocab: Dict[str, int], subtoken: bool, max_subtoken_num: int):
        """Turn asts to DGLgraphs. Require package 'dgl' and 'torch'.

        Args:
            asts: The input ast dict with form Dict[dir_name, Dict[file_name, Tree]] or Dict[idx, Tree](for jsonl format).
            token_vocab (Dict[str, int]): The input token dict where key is the token in ast, value is the token id.
            subtoken (bool): If split token into subtokens.
            max_subtoken_num (int): The largest number of subtokens in each token.
            mode (str): "source" for source files, "jsonl" for jsonl format.

        Returns:
            Dict[str, Dict[str, info_dict]]: The Graph dict with form Dict[dir_name, Dict[file_name, info_dict]],
                    info_dict is {"n_layers": int, "graph": dgl.DGLgraph, "node_types": List[str]}.

        Note:
            The edges direction in graph is from children to parents.
        """
        import torch
        from dgl import add_self_loop, graph

        def padding(input):
            if len(input) >= max_subtoken_num:
                return input[:max_subtoken_num]
            else:
                output = input + [0] * (max_subtoken_num - len(input))
                return output

        def gen_basic_graph(u, v, feats, node_types, node, vocab_dict):
            if subtoken:
                feat = torch.IntTensor(padding([vocab_dict[x] for x in node.token]))
            else:
                feat = torch.IntTensor([vocab_dict[node.token]])
            feats.append(feat)
            node_types.append(node.data.type)
            for child in node.children:
                v.append(node.id)
                u.append(child.id)
                gen_basic_graph(u, v, feats, node_types, child, vocab_dict)

        def gen_dgl_graph(ast):
            new_tree = PyAstParser.ast2any_tree(ast, subtoken=subtoken)
            n_layers = new_tree.height
            u, v, feats, node_types = [], [], [], []
            gen_basic_graph(u, v, feats, node_types, new_tree, token_vocab)
            g = graph((u, v))
            g.ndata["token_id"] = torch.stack(feats)
            g = add_self_loop(g)
            return n_layers, g, node_types

        print("Turn ast trees into DGLgraphs...")
        graphs = {}
        for dir_name in tqdm(asts, desc="transform trees to DGLgraphs..."):
            for f_name, ast in asts[dir_name].items():
                n_layers, g, node_types = gen_dgl_graph(ast)
                try:
                    graphs[dir_name][f_name] = {"n_layers": n_layers, "graph": g, "node_types": node_types}
                except Exception:
                    graphs[dir_name] = {f_name: {"n_layers": n_layers, "graph": g, "node_types": node_types}}

        print("finished!")

        return graphs

    @staticmethod
    def trees2DGLgraphs_for_probing(asts, token_vocab: Dict[str, int], p_batch: int = 1):
        """Turn asts to DGLgraphs. Require package 'dgl' and 'torch'.

        Args:
            asts: The input ast dict with form Dict[dir_name, Dict[file_name, Tree]] or Dict[idx, Tree](for jsonl format).
            token_vocab (Dict[str, int]): The input token dict where key is the token in ast, value is the token id.

        Returns:
            Dict[str, Dict[str, info_dict]]: The Graph dict with form Dict[dir_name, Dict[file_name, info_dict]],
                    info_dict is {"n_layers": int, "graph": dgl.DGLgraph, "node_types": List[str]}.

        Note:
            The edges direction in graph is from children to parents.
        """
        import torch
        from anytree import AnyNode
        from dgl import add_self_loop, graph

        def gen_basic_graph(u, v, feats, node_types, samples, node, vocab_dict):
            feat = torch.IntTensor([vocab_dict[node.token]])
            feats.append(feat)
            node_types.append(node.data.type)
            if len(samples) < p_batch:
                if node.id % step == 0:
                    id, infos = probing_sampler(node)
                    samples[id] = infos
            for child in node.children:
                v.append(node.id)
                u.append(child.id)
                gen_basic_graph(u, v, feats, node_types, samples, child, vocab_dict)

        def gen_dgl_graph(ast):
            new_tree = PyAstParser.ast2any_tree(ast)
            n_layers = new_tree.height
            ast_size = len(new_tree.descendants) + 1
            assert p_batch <= ast_size
            nonlocal step
            step = ast_size // p_batch  # the step for probing task sampler

            u, v, feats, node_types = [], [], [], []
            probing_samples = {}
            gen_basic_graph(u, v, feats, node_types, probing_samples, new_tree, token_vocab)
            assert len(probing_samples) == p_batch
            g = graph((u, v))
            g.ndata["token_id"] = torch.stack(feats)
            g = add_self_loop(g)
            return n_layers, g, node_types, probing_samples

        def to_class(n, type):
            # dist = {
            #     "size": [0, 165, 205, 250, 310, 450],
            #     "leaves": [0, 110, 135, 165, 205, 300],
            #     "not_leaves": [0, 56, 68, 82, 100, 150],
            # }
            dist = {
                "size": [0, 205, 310],
                "leaves": [0, 135, 205],
                "not_leaves": [0, 68, 100],
            }
            for idx, num in enumerate(dist[type]):
                if idx == len(dist[type]) - 1:
                    return idx
                if num < n <= dist[type][idx + 1]:
                    return idx

        def probing_sampler(node: AnyNode):
            height = node.height
            leaves = len(node.leaves)
            size = len(node.descendants) + 1
            not_leaves = size - leaves
            cl = to_class(leaves, "leaves")
            cs = to_class(size, "size")
            cnl = to_class(not_leaves, "not_leaves")
            return node.id, (height, cl, cnl, cs)

        print("Turn ast trees into DGLgraphs...")
        graphs = {}
        step = 0
        for dir_name in tqdm(asts, desc="transform trees to DGLgraphs..."):
            for f_name, ast in asts[dir_name].items():
                n_layers, g, node_types, p_samples = gen_dgl_graph(ast)
                try:
                    graphs[dir_name][f_name] = {"n_layers": n_layers, "graph": g, "node_types": node_types, "p_samples": p_samples}
                except Exception:
                    graphs[dir_name] = {f_name: {"n_layers": n_layers, "graph": g, "node_types": node_types, "p_samples": p_samples}}

        print("finished!")

        return graphs

    @staticmethod
    def trees2graphs(
        asts: Dict[str, Dict[str, Tree]],
        bidirectional_edge: bool = True,
        subtoken: bool = False,
        ast_only: bool = True,
        next_sib: bool = False,
        block_edge: bool = False,
        next_token: bool = False,
        next_use: bool = False,
        if_edge: bool = False,
        while_edge: bool = False,
        for_edge: bool = False,
    ):
        """Turn asts to graphs. Require package 'nextworkx' and 'anytree'.

        Args:
            asts (Dict[str, Dict[str, Tree]]): The input ast dict with form Dict[dir_name, Dict[file_name, Tree]].
            bidirectional_edge (bool, optional): If add bidirectional edge. Defaults to True.
            subtoken (bool): If True, tokens will be splited into list of subtokens.
            ast_only (bool, optional): If only build basic graph bases on origin ast. Defaults to True.
            next_sib (bool, optional): If add next sibling edge. Defaults to False.
            block_edge (bool, optional): If add next statement edge. Defaults to False.
            next_token (bool, optional): If add next token edge. Defaults to False.
            next_use (bool, optional): If add next use edge. Defaults to False.
            if_edge (bool, optional): If add IfStatement control flow edge. Defaults to False.
            while_edge (bool, optional): If add WhileStatement control flow edge. Defaults to False.
            for_edge (bool, optional): If add ForStatement control flow edge. Defaults to False.

        Returns:
            Dict[str, Dict[str, networkx.DiGraph]]: The Graph dict with form Dict[dir_name, Dict[file_name, DiGraph]].
        """
        from networkx import DiGraph

        def gen_basic_graph(node, graph):
            graph.add_node(node.id, token=node.token)
            for child in node.children:
                graph.add_node(node.id, token=node.token)
                graph.add_edge(node.id, child.id)
                if bidirectional_edge:
                    graph.add_edge(child.id, node.id)
                if not ast_only:
                    graph[node.id][child.id]["type"] = "Child"
                    graph[child.id][node.id]["type"] = "Parent"
                gen_basic_graph(child, graph)

        def gen_next_sib_edge(node, graph):
            for i in range(len(node.children) - 1):
                graph.add_edge(node.children[i].id, node.children[i + 1].id, type="NextSib")
                if bidirectional_edge:
                    graph.add_edge(node.children[i + 1].id, node.children[i].id, type="PrevSib")
            for child in node.children:
                gen_next_sib_edge(child, graph)

        def gen_next_stmt_edge(node, graph):
            token = node.token
            if token == "block":
                for i in range(len(node.children) - 1):
                    graph.add_edge(node.children[i].id, node.children[i + 1].id, type="NextStmt")
                    if bidirectional_edge:
                        graph.add_edge(node.children[i + 1].id, node.children[i].id, type="PrevStmt")
            for child in node.children:
                gen_next_stmt_edge(child, graph)

        def gen_next_token_edge(node, graph):
            def get_leaf_node_list(node, token_list):
                if len(node.children) == 0:
                    token_list.append(node.id)
                for child in node.children:
                    get_leaf_node_list(child, token_list)

            token_list = []
            get_leaf_node_list(node, token_list)
            for i in range(len(token_list) - 1):
                graph.add_edge(token_list[i], token_list[i + 1], type="NextToken")
                if bidirectional_edge:
                    graph.add_edge(token_list[i + 1], token_list[i], type="PrevToken")

        def gen_next_use_edge(node, graph):
            def get_vars(node, var_dict):
                if node.data.type == "identifier":
                    var = str(node.data.text)
                    if not var_dict.__contains__(var):
                        var_dict[var] = [node.id]
                    else:
                        var_dict[var].append(node.id)
                for child in node.children:
                    get_vars(child, var_dict)

            var_dict = {}
            get_vars(node, var_dict)
            for v in var_dict:
                for i in range(len(var_dict[v]) - 1):
                    graph.add_edge(var_dict[v][i], var_dict[v][i + 1], type="NextUse")
                    if bidirectional_edge:
                        graph.add_edge(var_dict[v][i + 1], var_dict[v][i], type="PrevUse")

        def gen_control_flow_edge(node, graph):
            token = node.token
            if while_edge:
                if token == "while_statement":
                    graph.add_edge(node.children[0].id, node.children[1].id, type="While")
                    graph.add_edge(node.children[1].id, node.children[0].id, type="While")
            if for_edge:
                if token == "for_statement":
                    for_type = PyAstParser.distinguish_for(node.data)
                    if for_type == "":
                        pass
                    elif for_type == "i":
                        graph.add_edge(node.children[0].id, node.children[1].id, type="For")
                        graph.add_edge(node.children[1].id, node.children[1].id, type="For")
                    elif for_type in {"c", "u"}:
                        graph.add_edge(node.children[0].id, node.children[1].id, type="For")
                        graph.add_edge(node.children[1].id, node.children[0].id, type="For")
                    elif for_type == "cu":
                        graph.add_edge(node.children[0].id, node.children[2].id, type="For")
                        graph.add_edge(node.children[1].id, node.children[0].id, type="For")
                        graph.add_edge(node.children[2].id, node.children[1].id, type="For")
                    elif for_type in {"ic", "iu"}:
                        graph.add_edge(node.children[0].id, node.children[1].id, type="For")
                        graph.add_edge(node.children[1].id, node.children[2].id, type="For")
                        graph.add_edge(node.children[2].id, node.children[1].id, type="For")
                    else:  # "icu"
                        graph.add_edge(node.children[0].id, node.children[1].id, type="For")
                        graph.add_edge(node.children[1].id, node.children[3].id, type="For")
                        graph.add_edge(node.children[2].id, node.children[1].id, type="For")
                        graph.add_edge(node.children[3].id, node.children[2].id, type="For")
                if token == "enhanced_for_statement":
                    graph.add_edge(node.children[0].id, node.children[1].id, type="For")
                    graph.add_edge(node.children[1].id, node.children[2].id, type="For")
                    graph.add_edge(node.children[1].id, node.children[3].id, type="For")
                    graph.add_edge(node.children[3].id, node.children[1].id, type="For")
            if if_edge:
                if token == "if_statement":
                    graph.add_edge(node.children[0].id, node.children[1].id, type="If")
                    if bidirectional_edge:
                        graph.add_edge(node.children[1].id, node.children[0].id, type="If")
                    if len(node.children) == 3:  # has else statement
                        graph.add_edge(node.children[0].id, node.children[2].id, type="IfElse")
                        if bidirectional_edge:
                            graph.add_edge(node.children[2].id, node.children[0].id, type="IfElse")
            for child in node.children:
                gen_control_flow_edge(child, graph)

        # print mode
        if ast_only:
            print("trees2graphs Mode: astonly")
        else:
            print(
                f"trees2graphs Mode: astonly=False, nextsib={next_sib}, ifedge={if_edge}, whileedge={while_edge}, foredge={for_edge}, blockedge={block_edge}, NextToken={next_token}, NextUse={next_use}"
            )

        graph_dict = {}
        for dir_name in tqdm(asts, desc="transform trees to graphs..."):
            for f_name, ast in asts[dir_name].items():
                new_tree = PyAstParser.ast2any_tree(ast, subtoken=subtoken)
                DiG = DiGraph()
                gen_basic_graph(new_tree, DiG)
                if not ast_only:
                    if next_sib:
                        gen_next_sib_edge(new_tree, DiG)
                    if block_edge:
                        gen_next_stmt_edge(new_tree, DiG)
                    if next_token:
                        gen_next_token_edge(new_tree, DiG)
                    if next_use:
                        gen_next_use_edge(new_tree, DiG)
                    gen_control_flow_edge(new_tree, DiG)
                try:
                    graph_dict[dir_name][f_name] = DiG
                except Exception:
                    graph_dict[dir_name] = {f_name: DiG}
        return graph_dict


if __name__ == "__main__":
    # f = "/home/qyh/projects/GTE/test/test.java"
    # parser = PyAstParser("/home/qyh/projects/GTE/utils/python-java-c-languages.so", "java")
    # ast = parser.file2ast(f)
    # anytree = PyAstParser.ast2any_tree(ast, subtoken=True)
    # print("over")
    def to_class(n, type):
        assert n > 0
        dist = {
            "size": [0, 165, 205, 250, 310, 450],
            "leaves": [0, 110, 135, 165, 205, 300],
            "not_leaves": [0, 56, 68, 82, 100, 150],
        }
        for idx, num in enumerate(dist[type]):
            if idx == len(dist[type]) - 1:
                return idx
            if num < n <= dist[type][idx + 1]:
                return idx

    i = to_class(0, "size")
    print(i)

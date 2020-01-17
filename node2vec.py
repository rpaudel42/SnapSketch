# ******************************************************************************
# node2vec.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 11/16/19   Paudel     Initial version,
# ******************************************************************************

import numpy as np
import random
import multiprocessing as mp

class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk_path = [G.node[start_node]['label']]
		walk = [start_node]
		# print(walk_length)
		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			# print("Current: ", cur," Label ", G.node[cur]['label'], "Neighbor: ", cur_nbrs)
			if len(cur_nbrs) > 0:
				# print("Alias: ", alias_nodes[cur])
				# print("Draw: ", alias_draw(alias_nodes[cur][0], alias_nodes[cur][1]))
				if len(walk) == 1:
					next = cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])]
					walk_path.append(G.node[next]['label'])
					# walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
					walk.append(next)
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
						alias_edges[(prev, cur)][1])]
					walk.append(next)
					walk_path.append(G.node[next]['label'])
			else:
				break
		# print("Start Node: ", G.node[start_node]['label'], "Walk: ", walk_path)
		return walk

	def simulate_walks(self, walk_length):
		# G = self.G
		# nodes = list(G.nodes())
		# random.shuffle(nodes)
		# pool = mp.Pool(mp.cpu_count())
		# walks =[pool.apply(self.node2vec_walk, args =(walk_length, node)) for node in nodes]
		# pool.close()
		G = self.G
		walks = []
		nodes = list(G.nodes())
		random.shuffle(nodes)

		for node in nodes:
			walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			try:
				all_edges = [G[dst][dst_nbr][i]['weight'] for i in range(0, len(G[dst][dst_nbr]))]
			except:
				all_edges = [1]
			if dst_nbr == src:							#remove [0] for other graph, only for multigraph
				unnormalized_probs.append(sum(all_edges)/ p)
				# unnormalized_probs.append(G[dst][dst_nbr][0]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				# unnormalized_probs.append(G[dst][dst_nbr][0]['weight'])
				unnormalized_probs.append(sum(all_edges))
			else:
				unnormalized_probs.append(sum(all_edges) / q)
				# unnormalized_probs.append(G[dst][dst_nbr][0]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = []
			# print("Node: ", node, sorted(G.neighbors(node)))
			# print(len(G[node][nbr]), G[node][nbr])    # remove [0] for other graph, only for multigraph
			for nbr in sorted(G.neighbors(node)):
				# print("Node: ", node, "Edge: ",  G[node][nbr])
				try:
					all_edges =  [G[node][nbr][i]['weight'] for i in range(0, len(G[node][nbr]))]
				except:
					all_edges = [1]


				# print("All edges: ", all_edges)
				unnormalized_probs.append(sum(all_edges))

			# unnormalized_probs = [G[node][nbr][0]['weight'] for nbr in sorted(G.neighbors(node))]

			# print("Un prob: ", node, unnormalized_probs)
			# print("Neighbor: ", len(G[node]), unnormalized_probs)
			norm_const = sum(unnormalized_probs)
			# print("Norm Const: ", norm_const)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			# print("Norm prob: ", node, normalized_probs)
			alias_nodes[node] = alias_setup(normalized_probs)
			# print("Alias: ", node, alias_nodes[node])

		alias_edges = {}
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges
		# print("Alias Node: ", alias_nodes)
		# print("Alias Edge: ", alias_edges)

		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
		q[kk] = K*prob
		if q[kk] < 1.0:
			smaller.append(kk)
		else:
			larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
		small = smaller.pop()
		large = larger.pop()

		J[small] = large
		q[large] = q[large] + q[small] - 1.0
		if q[large] < 1.0:
			smaller.append(large)
		else:
			larger.append(large)
	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
		return kk
	else:
		return J[kk]
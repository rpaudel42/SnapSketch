import networkx as nx
import random
import pandas as pd
import tqdm

class GraphUtils:

    def get_gex_graph(self, filename):
        G = nx.read_gexf(filename)
        return G

    def create_graph(self, graph):
        G = nx.MultiGraph() # make id di graph is 2 1 and 1 2 need to be different edge
        node ={}
        for n in graph['node']:
            node[n] = graph['node'][n]
            G.add_node(n, label = graph['node'][n])
        for e in graph['edge']:
            src, dest = e.split(' ')
            edge_count = graph['edge'][e]
            # # print("Edge:  ", graph['edge'][e])
            # if int(graph['edge'][e]) < 100:
            #     edge_count = int(graph['edge'][e])
            # else:
            #     edge_count = int(graph['edge'][e]/10)
            # for i in range(edge_count): #range(graph['edge'][e]):
            #     G.add_edge(src, dest, label = graph['edge'][e], weight=1)
            G.add_edge(src, dest, label=graph['edge'][e], weight=graph['edge'][e])
        return G

    def create_graph_duplicate_edge(self, graph):
        G = nx.MultiGraph()
        node ={}
        for n in graph['node']:
            node[n] = graph['node'][n]
            G.add_node(n, label = graph['node'][n])
        for e in graph['edge']:
            # print(e)
            for k, v in e.items():
                src, dest = k.split(' ')
                G.add_edge(src, dest, label = v, weight=v)
        return G

    def read_graph(self, file_name):
        '''
        PURPOSE: Read Synthetic Graph and load as dictionary of a graph file
        :param file_name:
        :param label:
        :return:
        '''
        graph = {}
        node = {}
        edge = {}
        label = 0
        with open(file_name) as f:
            lines = f.readlines()
        for l in lines:
            graph_entry = l.split(" ")
            if graph_entry[0] == 'XP':
                label = 0
            if graph_entry[0] == 'XN':
                label = 1

            if graph_entry[0] == 'v':
                node[graph_entry[1]] = graph_entry[2].strip('\n')

            if graph_entry[0] == 'd' or graph_entry[0] == 'u':
                edge[graph_entry[1] + ' ' + graph_entry[2].strip('\n')] = graph_entry[3].strip("\n")
        graph["node"] = node
        graph["edge"] = edge
        graph["label"] = label
        return graph

    def read_send_gfiles(self, fileName):

        '''PURPOSE: Read .G Graph, load each XP as JSON and send each XP as a graph stream
                :param fileName:
                :return:
        '''
        node_id = {}
        edge_id = {}
        graph = {}
        node = {}
        edge = {}
        nId = 1
        eId = 1
        g_list = {}
        XP = 0
        label = 'pos'
        anom_count = 0
        with open(fileName) as f:
            lines = f.readlines()
            for line in lines:
                singles = line.split(' ')
                if singles[0] == "XP" or singles[0] == "XN":
                    if len(singles)>1:
                        graph_id = singles[4].strip('\n')

                        # print("Graph: ", graph_id, "Nodes: ", node)
                        if singles[0] == "XP":
                            label = 0
                            anom_count = 0
                        if singles[0] == "XN":
                            label = 1
                            anom_count = 1
                        if XP > 0:
                            graph["node"] = node
                            graph["edge"] = edge
                            graph["label"] = label
                            graph["anom_count"] = anom_count
                            g_list[graph_id] = graph

                        graph = {}
                        node = {}
                        edge = {}
                        XP += 1
                elif singles[0] == "v":
                    # print("Nodes: ", line)
                    n = singles[2].strip('\n').strip('\"')
                    if n not in node_id:
                        node_id[n] = nId
                    nId += 1
                    node[singles[1]] = node_id[n]
                elif (singles[0] == "u" or singles[0] == "d"):
                    e = singles[3].strip('\n').strip('\"')
                    if e not in edge_id:
                        edge_id[e] = eId
                    eId += 1

                    edge[singles[1] + ' ' + singles[2]] = int(edge_id[e])
        return g_list

    def get_graph(self, filename):
        G=nx.Graph()
        f=open(filename,'r')
        lines=f.readlines()
        for line in lines:
            if(line[0]=='#'):
                continue
            else:
                temp=line.split()
                index1=int(temp[0])
                index2=int(temp[1])
                G.add_edge(index1,index2)
        f.close()
        return G

    def get_graph_from_csv(self, csv_file):
        print("\n\n ---- Creating G Files -----")
        firewall_log = pd.DataFrame(index=[], columns=[])
        log = pd.read_csv(csv_file)
        firewall_log = firewall_log.append(log)
        firewall_log = firewall_log.iloc[:, [1, 2, 3, 4]]
        firewall_log.columns = ['source', 'destination', 'anomaly', 'time_past']
        firewall_log = firewall_log[firewall_log.source != '(empty)'].reset_index()
        #print(firewall_log)

        g_list = {}
        graph = {}
        node = {}
        edge = []
        label = 0

        global_nodes = {}
        local_node = {}
        global_node_id = 1
        local_node_id = 1
        hour = 0
        anom_count = 0
        for index, row in firewall_log.iterrows():
            if 1 == 1: #row['time_past'] < 5: #1 == 1: #
                if row['source'] not in global_nodes:
                    global_nodes[row['source']] = global_node_id
                    global_node_id += 1

                if row['destination'] not in global_nodes:
                    global_nodes[row['destination']] = global_node_id
                    global_node_id += 1

                curr_hour = row['time_past']

                if hour != curr_hour:
                    print("\n\n Hour Past: ", hour, "    ", index)
                    graph['node'] = node
                    # edge_list = []
                    # for e in edge:
                    #     print (e)
                    graph['edge'] = edge
                    graph['anom_count'] = anom_count
                    if anom_count >= 50:
                        graph['label'] = 1
                    else:
                        graph['label'] = 0
                    g_list[hour] = graph

                    graph = {}
                    node = {}
                    edge = []
                    local_node = {}
                    local_node_id = 1
                    anom_count = 0

                hour = row['time_past']

                if row['anomaly'] == 1:
                    anom_count += 1

                if row['source'] not in local_node:
                    local_node[row['source']] = local_node_id  # global_nodes[row['source']]
                    node[local_node_id] = global_nodes[row['source']]
                    local_node_id += 1

                if row['destination'] not in local_node:
                    node[local_node_id] =global_nodes[row['destination']]
                    local_node[row['destination']] = local_node_id  # global_nodes[row['destination']]
                    local_node_id += 1

                edge_id = str(local_node[row['source']])+ ' ' +str(local_node[row['destination']])
                e = {}
                e[edge_id] = 1
                edge.append(e) # edge and edge weight
                # if edge_id in edge:
                #     count = edge[edge_id]
                #     edge[edge_id] = count + 1
                # else:
                #     edge[edge_id] = 1
        return g_list

    def create_taxi_graph(self, csv_file):
        print("\n\n ---- Creating Taxi Graphs -----")
        taxi_log = pd.DataFrame(index=[], columns=[])
        log = pd.read_csv(csv_file)
        taxi_log = taxi_log.append(log)
        taxi_log = taxi_log.iloc[:, [1, 2, 3, 4, 5]]
        taxi_log.columns = ['source', 'destination', 'time_past', 'anomaly','cur_time']
        taxi_log = taxi_log[taxi_log.source != '(empty)'].reset_index()

        print(taxi_log)

        g_list = {}
        graph = {}
        node = {}
        edge = {}
        label = 0

        global_nodes = {}
        local_node = {}
        global_node_id = 1
        local_node_id = 1
        hour = 0
        anom_count = 0
        total_edge = 0
        graph_id = 1
        for index, row in taxi_log.iterrows():
            if 1 == 1:  # row['hours_past'] < 5:
                if row['source'] not in global_nodes:
                    global_nodes[row['source']] = global_node_id
                    global_node_id += 1

                if row['destination'] not in global_nodes:
                    global_nodes[row['destination']] = global_node_id
                    global_node_id += 1

                curr_hour = row['time_past']
                curr_time = row['cur_time']

                if hour != curr_hour:
                    print("\n\n Hour Past: ", hour, "    ", index)
                    graph['node'] = node
                    graph['edge'] = edge
                    graph['anom_count'] = total_edge  # anom_count
                    graph['g_time'] = curr_time
                    if anom_count >= 100:
                        graph['label'] = 1
                    else:
                        graph['label'] = 0
                    g_list[str(graph_id)] = graph

                    graph = {}
                    node = {}
                    edge = {}
                    local_node = {}
                    local_node_id = 1
                    total_edge = 0
                    graph_id += 1

                hour = row['time_past']

                if row['anomaly'] == 1:
                    anom_count += 1

                if row['source'] not in local_node:
                    local_node[row['source']] = local_node_id  # global_nodes[row['source']]
                    node[local_node_id] = global_nodes[row['source']]
                    local_node_id += 1

                if row['destination'] not in local_node:
                    node[local_node_id] = global_nodes[row['destination']]
                    local_node[row['destination']] = local_node_id  # global_nodes[row['destination']]
                    local_node_id += 1
                edge_id = str(local_node[row['source']]) + ' ' + str(local_node[row['destination']])
                total_edge += 1
                if edge_id in edge:
                    count = edge[edge_id]
                    edge[edge_id] = count + 1
                else:
                    edge[edge_id] = 1
        return g_list

    dictionary = {"10.200.150.1": "FWtoInternet", "172.20.1.1": "FWtoEWS", "172.20.1.5": "ExternalWeb",
                  "192.168.1.16": "IDS", "192.168.1.1": "FWtoDataVLAN",
                  "192.168.2.1": "FWtoOfficeVLAN",
                  "192.168.1.2": "DHCP", "192.168.1.3": "HRDB", "192.168.1.4": "ShDB",
                  "192.168.1.5": "InternalWeb", "192.168.1.6": "MailServer", "192.168.1.7": "FileServer",
                  "192.168.1.14": "DNS", "192.168.1.50": "FirewallLog"}

    def create_vast_graph(self, csv_file):
        print("\n\n ---- Creating G Files -----")
        firewall_log = pd.DataFrame(index=[], columns=[])
        log = pd.read_csv(csv_file)
        firewall_log = firewall_log.append(log)
        print("Here: ", firewall_log)

        firewall_log = firewall_log.iloc[:, [1, 2, 3, 4, 5]]
        firewall_log.columns = ['source', 'destination', 'time_past', 'anomaly']
        firewall_log = firewall_log[firewall_log.source != '(empty)'].reset_index()
        print(firewall_log)

        g_list = {}
        graph = {}
        node = {}
        edge = {}
        label = 0

        global_nodes = {}
        local_node = {}
        global_node_id = 1
        local_node_id = 1
        hour = 0
        anom_count = 0
        for index, row in firewall_log.iterrows():
            if 1 == 1:  # row['hours_past'] < 5:
                if row['source'] not in global_nodes:
                    global_nodes[row['source']] = global_node_id
                    global_node_id += 1

                if row['destination'] not in global_nodes:
                    global_nodes[row['destination']] = global_node_id
                    global_node_id += 1

                curr_hour = row['time_past']

                if hour != curr_hour:
                    print("\n\n Hour Past: ", hour, "    ", index)
                    graph['node'] = node
                    graph['edge'] = edge
                    graph['anom_count'] = anom_count
                    if anom_count >= 100:
                        graph['label'] = 1
                    else:
                        graph['label'] = 0
                    g_list[str(hour)] = graph

                    graph = {}
                    node = {}
                    edge = {}
                    local_node = {}
                    local_node_id = 1
                    anom_count = 0

                hour = row['time_past']

                if row['anomaly'] == 1:
                    anom_count += 1

                if row['source'] not in local_node:
                    local_node[row['source']] = local_node_id  # global_nodes[row['source']]
                    node[local_node_id] = global_nodes[row['source']]
                    local_node_id += 1

                if row['destination'] not in local_node:
                    node[local_node_id] = global_nodes[row['destination']]
                    local_node[row['destination']] = local_node_id  # global_nodes[row['destination']]
                    local_node_id += 1
                edge_id = str(local_node[row['source']]) + ' ' + str(local_node[row['destination']])
                if edge_id in edge:
                    count = edge[edge_id]
                    edge[edge_id] = count + 1
                else:
                    edge[edge_id] = 1
        return g_list

    def get_graph_from_csv_weighted(self, csv_file):
        print("\n\n ---- Creating G Files -----")
        firewall_log = pd.DataFrame(index=[], columns=[])
        log = pd.read_csv(csv_file)
        firewall_log = firewall_log.append(log)
        print("Here: ", firewall_log)

        firewall_log = firewall_log.iloc[:, [1, 2, 3, 4, 5]]
        firewall_log.columns = ['source', 'destination', 'time_past', 'anomaly']
        firewall_log = firewall_log[firewall_log.source != '(empty)'].reset_index()
        print(firewall_log)

        g_list = {}
        graph = {}
        node = {}
        edge = {}
        label = 0

        global_nodes = {}
        local_node = {}
        global_node_id = 1
        local_node_id = 1
        hour = 0
        anom_count = 0
        for index, row in firewall_log.iterrows():
            if 1 == 1:  # row['hours_past'] < 5:
                if row['source'] not in global_nodes:
                    global_nodes[row['source']] = global_node_id
                    global_node_id += 1

                if row['destination'] not in global_nodes:
                    global_nodes[row['destination']] = global_node_id
                    global_node_id += 1

                curr_hour = row['time_past']

                if hour != curr_hour:
                    print("\n\n Hour Past: ", hour, "    ", index)
                    graph['node'] = node
                    graph['edge'] = edge
                    graph['anom_count'] = anom_count
                    if anom_count >= 100:
                        graph['label'] = 1
                    else:
                        graph['label'] = 0
                    g_list[str(hour)] = graph

                    graph = {}
                    node = {}
                    edge = {}
                    local_node = {}
                    local_node_id = 1
                    anom_count = 0

                hour = row['time_past']

                if row['anomaly'] == 1:
                    anom_count += 1

                if row['source'] not in local_node:
                    local_node[row['source']] = local_node_id  # global_nodes[row['source']]
                    node[local_node_id] = global_nodes[row['source']]
                    local_node_id += 1

                if row['destination'] not in local_node:
                    node[local_node_id] =global_nodes[row['destination']]
                    local_node[row['destination']] = local_node_id  # global_nodes[row['destination']]
                    local_node_id += 1
                edge_id = str(local_node[row['source']])+ ' ' +str(local_node[row['destination']])
                if edge_id in edge:
                    count = edge[edge_id]
                    edge[edge_id] = count + 1
                else:
                    edge[edge_id] = 1
        return g_list

    def get_stats(self, G):
        stats ={}
        stats['num_nodes'] = nx.number_of_nodes(G)
        stats['num_edges'] = nx.number_of_edges(G)
        stats['is_Connected'] = nx.is_connected(G)

    def draw_graph(self, G, index):
        pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos)
        plt.savefig("data/pdfs/"+str(index)+".pdf")
        #plt.show()


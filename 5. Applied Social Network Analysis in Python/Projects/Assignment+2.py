
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.2** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-social-network-analysis/resources/yPcBs) course resource._
# 
# ---

# # Assignment 2 - Network Connectivity
# 
# In this assignment you will go through the process of importing and analyzing an internal email communication network between employees of a mid-sized manufacturing company. 
# Each node represents an employee and each directed edge between two nodes represents an individual email. The left node represents the sender and the right node represents the recipient.

# In[1]:


import networkx as nx

# This line must be commented out when submitting to the autograder
#!head email_network.txt


# ### Question 1
# 
# Using networkx, load up the directed multigraph from `email_network.txt`. Make sure the node names are strings.
# 
# *This function should return a directed multigraph networkx graph.*

# In[2]:


def answer_one():
    
    G = nx.read_edgelist('email_network.txt', data=[('Time', int)], nodetype=str, create_using=nx.MultiDiGraph())
    
    return G
answer_one()


# ### Question 2
# 
# How many employees and emails are represented in the graph from Question 1?
# 
# *This function should return a tuple (#employees, #emails).*

# In[3]:


def answer_two():
        
    G = answer_one()
    
    return len(G.nodes()), len(G.edges())
answer_two()


# ### Question 3
# 
# * Part 1. Assume that information in this company can only be exchanged through email.
# 
#     When an employee sends an email to another employee, a communication channel has been created, allowing the sender to provide information to the receiver, but not vice versa. 
# 
#     Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?
# 
# 
# * Part 2. Now assume that a communication channel established by an email allows information to be exchanged both ways. 
# 
#     Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?
# 
# 
# *This function should return a tuple of bools (part1, part2).*

# In[4]:


def answer_three():
     
    G1 = answer_one()
    
    G2 = G1.to_undirected()
    
    part1 = nx.is_strongly_connected(G1)
    
    part2 = nx.is_connected(G2)
    
    return part1, part2
answer_three()


# ### Question 4
# 
# How many nodes are in the largest (in terms of nodes) weakly connected component?
# 
# *This function should return an int.*

# In[5]:


def answer_four():
    
    import numpy as np
        
    G = answer_one()
    
    wcc = sorted(nx.weakly_connected_components(G))
    
    wccl = [len(c) for c in wcc]
    
    order = np.argsort(wccl)
    
    return len(wcc[order[-1]])
answer_four()


# ### Question 5
# 
# How many nodes are in the largest (in terms of nodes) strongly connected component?
# 
# *This function should return an int*

# In[6]:


def answer_five():   
    
    import numpy as np
        
    G = answer_one()
    
    scc = sorted(nx.strongly_connected_components(G))
    
    sccl = [len(c) for c in scc]
    
    order = np.argsort(sccl)
    
    return len(scc[order[-1]])
answer_five()


# ### Question 6
# 
# Using the NetworkX function strongly_connected_component_subgraphs, find the subgraph of nodes in a largest strongly connected component. 
# Call this graph G_sc.
# 
# *This function should return a networkx MultiDiGraph named G_sc.*

# In[7]:


def answer_six():
    
    G = answer_one()
    
    sc_subgraphs = nx.strongly_connected_component_subgraphs(G)
    
    G_sc = max(sc_subgraphs, key=len)
    
    G_sc.name = 'G_sc'
    
    return G_sc
answer_six()


# ### Question 7
# 
# What is the average distance between nodes in G_sc?
# 
# *This function should return a float.*

# In[8]:


def answer_seven():
        
    G = answer_six()
    
    return nx.average_shortest_path_length(G)
answer_seven()


# ### Question 8
# 
# What is the largest possible distance between two employees in G_sc?
# 
# *This function should return an int.*

# In[9]:


def answer_eight():
        
    G = answer_six()
    
    return nx.diameter(G)
answer_eight()


# ### Question 9
# 
# What is the set of nodes in G_sc with eccentricity equal to the diameter?
# 
# *This function should return a set of the node(s).*

# In[10]:


def answer_nine():
       
    G = answer_six()
    
    return set(nx.periphery(G))
answer_nine()


# ### Question 10
# 
# What is the set of node(s) in G_sc with eccentricity equal to the radius?
# 
# *This function should return a set of the node(s).*

# In[11]:


def answer_ten():
        
    G = answer_six()
    
    return set(nx.center(G))
answer_ten()


# ### Question 11
# 
# Which node in G_sc is connected to the most other nodes by a shortest path of length equal to the diameter of G_sc?
# 
# How many nodes are connected to this node?
# 
# 
# *This function should return a tuple (name of node, number of satisfied connected nodes).*

# In[12]:


def answer_eleven():
        
    G = answer_six()
    
    diameter = answer_eight()
    
    path_lenght_diameter = {node: list(nx.shortest_path_length(G, node).values()).count(diameter) for node in G.nodes()}
    
    return max(path_lenght_diameter.items(), key=lambda x: x[1])
answer_eleven()


# ### Question 12
# 
# Suppose you want to prevent communication from flowing to the node that you found in the previous question from any node in the center of G_sc, what is the smallest number of nodes you would need to remove from the graph (you're not allowed to remove the node from the previous question or the center nodes)? 
# 
# *This function should return an integer.*

# In[14]:


def answer_twelve():
    
    G = answer_six()
    
    center = list(answer_ten())[0]
    
    target = answer_eleven()[0]
        
    min_nodes_removing = len(nx.minimum_node_cut(G, center, target))
    
    return min_nodes_removing
answer_twelve()


# ### Question 13
# 
# Construct an undirected graph G_un using G_sc (you can ignore the attributes).
# 
# *This function should return a networkx Graph.*

# In[15]:


def answer_thirteen():
        
    G_sc = answer_six()
    
    G_un = nx.Graph(G_sc)
    
    G_un.name = 'G_un'
    
    return G_un
answer_thirteen()


# ### Question 14
# 
# What is the transitivity and average clustering coefficient of graph G_un?
# 
# *This function should return a tuple (transitivity, avg clustering).*

# In[16]:


def answer_fourteen():
        
    G = answer_thirteen()
    
    transitivity = nx.transitivity(G)
    
    avg_cc = nx.average_clustering(G)
    
    return transitivity, avg_cc
answer_fourteen()


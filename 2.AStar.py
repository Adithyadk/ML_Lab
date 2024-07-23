{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4d6f7ada",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Path found: ['S', 'A', 'C', 'G']\n",
      "Cost: 6\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "['S', 'A', 'C', 'G']"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 2. A-Star Algorithm\n",
    "\n",
    "def h(n):\n",
    "    H = {'S': 7,'A': 6,'B': 2,'C': 1,'D': 3,'G': 0}\n",
    "    return H[n]\n",
    "\n",
    "def a_star_algorithm(graph, start, goal):\n",
    "\n",
    "    open_list = [start]\n",
    "    closed_list = set()\n",
    "\n",
    "    g = {start:0}\n",
    "\n",
    "    parents = {start:start}\n",
    "\n",
    "    while open_list:\n",
    "\n",
    "        open_list.sort(key=lambda v: g[v] + h(v), reverse=True)\n",
    "        n = open_list.pop()\n",
    "\n",
    "        # If node is goal then construct the path and return\n",
    "        if n == goal:\n",
    "            reconst_path = []\n",
    "\n",
    "            while parents[n] != n:\n",
    "                reconst_path.append(n)\n",
    "                n = parents[n]\n",
    "\n",
    "            reconst_path.append(start)\n",
    "            reconst_path.reverse()\n",
    "\n",
    "            print(f'Path found: {reconst_path}')\n",
    "            print(f'Cost: {g[goal]}')   # Cost Printing\n",
    "            return reconst_path\n",
    "\n",
    "        for (m, weight) in graph[n]:\n",
    "        # if m is first visited, add it to open_list and note its parent\n",
    "            if m not in open_list and m not in closed_list:\n",
    "                open_list.append(m)\n",
    "                parents[m] = n\n",
    "                g[m] = g[n] + weight\n",
    "\n",
    "            # otherwise, check if it's quicker to first visit n, then m\n",
    "            # and if it is, update parent and g data\n",
    "            # and if the node was in the closed_list, move it to open_list\n",
    "            else:\n",
    "                if g[m] > g[n] + weight:\n",
    "                    g[m] = g[n] + weight\n",
    "                    parents[m] = n\n",
    "\n",
    "                    if m in closed_list:\n",
    "                        closed_list.remove(m)\n",
    "                        open_list.append(m)\n",
    "\n",
    "        # Node's neighbours are visited. Now put node to closed list.\n",
    "        closed_list.add(n)\n",
    "\n",
    "    print('Path does not exist!')\n",
    "    return None\n",
    "\n",
    "\n",
    "graph = {\n",
    "    'S': [('A', 1), ('B', 4)],\n",
    "    'A': [('C', 2), ('D', 5)],\n",
    "    'B': [('D', 1)],\n",
    "    'C': [('G', 3)],\n",
    "    'D': [('G', 2)],\n",
    "    'G': []  # Goal node\n",
    "}\n",
    "\n",
    "a_star_algorithm(graph, 'S', 'G')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "69af0f6c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

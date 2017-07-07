/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   divide_dataset.c                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: pmclaugh <pmclaugh@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2017/07/06 18:34:31 by pmclaugh          #+#    #+#             */
/*   Updated: 2017/07/06 21:49:09 by pmclaugh         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "dispatcher.h"
#include <math.h>
#include <limits.h>
#define SORT_NAME mort
#define SORT_TYPE t_sortbod
#define SORT_CMP(x, y) (sbod_comp(&x, &y))
#include "sort.h"

/*
    Theta is a variable in the Barnes-Hut algorithm.
    It's the threshold that determines if a collection of particles is
    far enough away to approximate as one massive body at their center of gravity.

    Lowering it increases computational load but reduces error.
    At 0, the algorithm degenerates to an all-pairs O(n^2) calculation.
*/
#define THETA 1

/*
    Leaf threshold determines how finely we divide the space.
    If a cell has more than leaf_threshold particles, it gets subdivided.

    In the basic Barnes-Hut algorithm, leaf threshold is 1.
    The distributed nature of our approach made this impractical and unnecessary.
    This is discussed further at the multipole acceptance function below
*/
#define LEAF_THRESHOLD pow(2, 12)

/*
    Divide_dataset is the heart of the dispatcher program.
    
    It partitions the set of particles into an octree, and
    uses that structure to divide the work of calculating a frame of the simulation.
    We use a fast method that reduces this to a sorting problem.
    We sort the particles by Morton encodings of their positions and this implicitly orders them in an octree.
    
    Once the tree is generated, each cell needs to determine its "neighborhood", which is
    the collection of particles that will be needed to accurately calculate the forces on the particles in the cell.

    This generates many "workunits", which we send to the worker computers in collections called "bundles".
    
    This is the single-threaded version of divide_dataset.c, used during development.
    In practice each step was multithreaded, but I think this is easier to read.
*/

void    divide_dataset(t_dispatcher *dispatcher)
{
    static t_tree *t;

    if (t != NULL)
        free_tree(t); //the tree from last frame is needed until the moment we're making the new one

    t = make_tree(dispatcher->dataset->particles, dispatcher->dataset->particle_cnt);

    t_tree **leaves = enumerate_leaves(t);

    for (int i = 0; leaves[i]; i++)
        leaves[i]->neighbors = assemble_neighborhood(leaves[i], t);

    bundle_all(dispatcher, leaves);
}

t_tree *make_tree(t_body *bodies, int count)
{
    //determine bounding cube of particle set
    t_bounds root_bounds = bounds_from_bodies(bodies, count);
    
    //generate and cache morton codes for each body
    t_sortbod *sorts = make_sortbods(bodies, root_bounds, count);
    
    //sort the bodies by their morton codes
    //they are now arranged on a z-order curve.
    mort_tim_sort(sorts, count);
    
    //copy data back from cached sort structure
    uint64_t *mortons = calloc(count, sizeof(uint64_t));
    for (int i = 0; i < count; i++)
    {
        bodies[i] = sorts[i].bod;
        mortons[i] = sorts[i].morton;;
    }

    t_tree *root = new_tnode(bodies, count, NULL);
    root->bounds = root_bounds;
    root->mortons = mortons;
    
    //recursively divide the tree
    split_tree(root);
    free(sorts);
    free(mortons);
    return (root);
}

///////////////////////////////////////////////////////////////
///////////////       Morton Codes             ///////////////
///////////////////////////////////////////////////////////////

// fast method to insert 2 0's before each bit of an unsigned int,
// ie 1101 becomes 001001000001
// credit to http://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
uint64_t splitBy3(const unsigned int a)
{
    uint64_t x = a & 0x1fffff; // we only look at the first 21 bits
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}
 
uint64_t mortonEncode_magicbits(const unsigned int x, const unsigned int y, const unsigned int z)
{
    //interweave the rightmost 21 bits of 3 unsigned ints to generate a 63-bit morton code
    uint64_t answer = 0;
    answer |= splitBy3(z) | splitBy3(y) << 1 | splitBy3(x) << 2;
    return answer;
}

//adapted from NVIDIA's 10bit version
uint64_t morton64(float x, float y, float z)
{
    //x, y, z are in [0, 1]. multiply by 2^21 to get 21 bits, confine to [0, 2^21 - 1]
    x = fmin(fmax(x * 2097152.0f, 0.0f), 2097151.0f);
    y = fmin(fmax(y * 2097152.0f, 0.0f), 2097151.0f);
    z = fmin(fmax(z * 2097152.0f, 0.0f), 2097151.0f);
    return (mortonEncode_magicbits((unsigned int)x, (unsigned int)y, (unsigned int)z));
}


t_sortbod *make_sortbods(t_body *bodies, t_bounds bounds, int count)
{
    t_sortbod *sorts = calloc(count, sizeof(t_sortbod));
    float distance =  1.0 / (bounds.xmax - bounds.xmin);
    for (int i = 0; i < count; i++)
    {
        sorts[i].morton = morton64((bodies[i].position.x - bounds.xmin) * distance, (bodies[i].position.y - bounds.ymin) * distance, (bodies[i].position.z - bounds.zmin) * distance);
        sorts[i].bod = bodies[i];
    }
    return (sorts);
}

///////////////////////////////////////////////////////////////
///////////////       Making The Tree           ///////////////
///////////////////////////////////////////////////////////////

/*
    When the particles are sorted in Morton order, they are implicitly arranged in an octree;
    We just need to identify the borders between cells.

    The Morton codes are 63 bits, 21 3-bit subcodes. They're like 21-digit numbers in base 8.
    Example with 12 bits / 4 subcodes, * is any digit:
    -The root contains all the particles ****
    -Its children are the 8 sets 0***, 1***, 2***...
    -The children of 0*** are the sets like 00**, 01**, 02**, etc.

    Since it's already a sorted list, we do a binary search for the digit borders and construct the tree.
*/

int binary_border_search(uint64_t *mortons, int startind, int maxind, unsigned int code, int depth)
{
    if (startind == maxind)
        return 0;//empty cell at end of parent
    uint64_t m = mortons[startind];
    m = m << (1 + 3 * depth);
    m = m >> 61;
    if (m != code)
        return 0; // empty cell.
    int step = (maxind - startind) / 2;
    int i = startind;
    while (i < maxind - 1)
    {
        m = mortons[i];
        m = m << (1 + 3 * depth);
        m = m >> 61;
        uint64_t mnext = mortons[i + 1];
        mnext = mnext << (1 + 3 * depth);
        mnext = mnext >> 61;
        if (m == code && mnext != code)
            return (i - startind + 1); //found border
        else if (m == code)
            i += step;//step forward
        else
            i -= step;//step backward
        step /= 2;
        if (step == 0)
            step = 1;
    }
    return (maxind - startind);
}

void split(t_tree *node)
{
    //split this cell into 8 octants
    node->children = (t_tree **)calloc(8, sizeof(t_tree *));
    int depth = node_depth(node);
    unsigned int offset = 0;
    for (unsigned int i = 0; i < 8; i++)
    {
        node->children[i] = (t_tree *)calloc(1, sizeof(t_tree));
        node->children[i]->bodies = &(node->bodies[offset]);
        node->children[i]->mortons = &(node->mortons[offset]);
        node->children[i]->count = 0;
        node->children[i]->parent = node;
        node->children[i]->children = NULL;
        node->children[i]->bounds = bounds_from_code(node->bounds, i);
        
        //scan through array for borders between 3-bit substring values for this depth
        //these are the dividing lines between octants
        unsigned int j = binary_border_search(node->mortons, offset, node->count, i, depth);
        offset += j;
        node->children[i]->count = j;
    }
}

void split_tree(t_tree *root)
{
    //we divide the tree until each leaf cell has < leaf threshold bodies.
    //we also have to halt at depth 21, but it is unlikely to need to divide that far.
    if (root->count < LEAF_THRESHOLD || node_depth(root) == 21)
    {
        root->as_single = make_as_single(root);
        return;
    }
    split(root);
    for (int i = 0; i < 8; i++)
        split_tree(root->children[i]);
    root->as_single = make_as_single(root);
}

/*
    Return a t_tree** that's all the leaf nodes (ie childless nodes) in the tree
    This is an opportunity to very quickly do centers of gravity/as_single

    This (and the other traversing functions) might be better implemented using a stack and queue, 
    but in practice this operation is quite quick anyway.
*/

static t_tree **enumerate_leaves(t_tree *root)
{
    t_tree **ret;

    if (!root->children)
    {
        ret = (t_tree **)calloc(2, sizeof(t_tree *));
        ret[0] = root->count ? root : NULL; //we do not bother enumerating empty leaves
        ret[1] = NULL;
        return (ret);
    }
    t_tree ***returned = (t_tree ***)calloc(8, sizeof(t_tree **));
    int total = 0;
    for (int i = 0; i < 8; i++)
    {
        returned[i] = enumerate_leaves(root->children[i]);
        total += count_tree_array(returned[i]);
    }
    ret = (t_tree **)calloc(total + 1, sizeof(t_tree *));
    for (int i = 0; i < total;)
    {
        for (int j = 0; j < 8; j++)
        {
            for (int k = 0; returned[j][k]; k++, i++)
            {
                ret[i] = returned[j][k];
            }
            free(returned[j]);
        }
        free(returned);
    }
    ret[total] = NULL;
    return (ret);
}

///////////////////////////////////////////////////////////////
///////////////   Neighborhoods and Barnes-Hut  ///////////////
///////////////////////////////////////////////////////////////

/*
    Recursively flow through the tree, determining if cells are "near" or "far" from
    the cell we're currently considering. We skip the root.
    
    If the cell is far away (m_a_c < THETA), that cell is far enough away to treat as 1 particle.
    we return the as_single version of the cell.
    
    If the cell is nearby and childless (ie leaf), it is near enough that direct calculation is necessary,
    we return the normal, many-particle version of the cell.

    If the cell is nearby and has children, we recurse down to its children.

    In this way, we enumerate the "Neighborhood" of the cell, 
    which encompasses all the bodies we'll need to compare with on the GPU.
*/

static t_tree **assemble_neighborhood(t_tree *cell, t_tree *root)
{
    t_tree **ret;
    t_tree ***returned;

    

    if (root->parent && multipole_acceptance_criterion(cell, root) < THETA)
    {
        ret = (t_tree **)calloc(2, sizeof(t_tree *));
        ret[0] = root->as_single;
        ret[1] = NULL;
        return (ret);
    }
    else if (!(root->children))
    {
        ret = (t_tree **)calloc(2, sizeof(t_tree *));
        ret[0] = root;
        ret[1] = NULL;
        return (ret);
    }
    else
    {
        returned = (t_tree ***)calloc(8, sizeof(t_tree **));
        int total = 0;
        for (int i = 0; i < 8; i++)
        {
            returned[i] = assemble_neighborhood(cell, root->children[i]);
            total += count_tree_array(returned[i]);
        }
        ret = (t_tree **)calloc(total + 1, sizeof(t_tree *));
        for (int i = 0; i < total;)
        {
            for (int j = 0; j < 8; j++)
            {
                if (!returned[j])
                    continue ;
                for (int k = 0; returned[j][k]; k++, i++)
                {
                    ret[i] = returned[j][k];
                }
                free(returned[j]);
            }
            free(returned);
        }
        ret[total] = NULL;
        return (ret);
    }
}

static float multipole_acceptance_criterion(t_tree *us, t_tree *them)
{
    /*
        assess whether a cell is "near" or "far" for the sake of barnes-hut algorithm
        if the value returned is less than THETA, that cell is "far"
    */

    float s;
    float d;
    cl_float4 r;
    cl_float4 us_midpoint;

    if (us == them)
        return (THETA);
    us_midpoint = midpoint_from_bounds(us->bounds);
    s = them->bounds.xmax - them->bounds.xmin;
    r.x = them->as_single->bodies[0].position.x - us_midpoint.x;
    r.y = them->as_single->bodies[0].position.y - us_midpoint.y;
    r.z = them->as_single->bodies[0].position.z - us_midpoint.z;
    d = sqrt(r.x * r.x + r.y * r.y + r.z * r.z);

    /*
        In normal Barnes-Hut, the MAC is evaluated for every body vs every cell. 
        We're evaluating it cell to cell for better distribution and parallelism.
        This could result in some comparisons that should be "near" being done as "far".
        Subtracting half of our cell's width from the measured distance compensates for this adequately.
        Basically, we are evaluating as if all the bodies in our cell are right up against the nearest side of our cell.

        This may result in some added computational load, as we are essentially lowering theta.
        However, homogenizing the particles' neighborhoods allows for good parallelism on the GPU,
        and makes distributing the work easier.
    */

    d -= (us->bounds.xmax - us->bounds.xmin) / 2;
    return (s/d);
}


///////////////////////////////////////////////////////////////
///////////////            Bundling             ///////////////
///////////////////////////////////////////////////////////////

/*
    Each leaf cell and its neighborhood, which we call a workunit, needs to be sent
    to the workers for computation. This would naively result in sending a total amount of data 
    that's a large multiple of the dataset size. However, neighborhoods of nearby cells can have very high overlap.

    We use a specialized hash table to track which cells we've seen already, and which workunits they belong to. (hash.c)
    The table is then serialized as a "bundle", which is unpacked on the worker end as workunits.
    By bundling them in the order they were enumerated, we maximize overlap (the array of leaves is Z-ordered)

    The elimination of sending redundant workunit data reduced network load by 10-20x.
*/

t_bundle *bundle_leaves(t_tree **leaves, int offset, int count)
{
    t_dict *dict = create_dict(1000);
    
    t_pair *ids = NULL;
    long difficulty = 0;
    long size = 0;
    for (int i = 0; i < count && leaves[offset + i]; i++)
    {
        if (!ids)
            ids = create_pair(i + offset);
        else
        {
            t_pair *p = create_pair(i + offset);
            p->next_key = ids;
            ids = p;
        }
        size += leaves[i]->count * sizeof(t_body);
        t_tree **adding = leaves[offset + i]->neighbors;
        for (int j = 0; adding[j]; j++)
        {
            dict_insert(dict, adding[j], i);
            difficulty += leaves[offset + i]->count * adding[j]->count;
            size += adding[j]->count * sizeof(cl_float4);
        }
    }
    t_bundle *b = bundle_dict(dict, ids);
    b->difficulty = difficulty;
    b->size = size;
    return (b);
}

t_msg serialize_bundle(t_bundle *b, t_tree **leaves)
{
    t_msg m;
    m.size = sizeof(int) * 2;
    for (int i = 0; i < b->keycount; i++)
    {
        m.size += sizeof(int) * 2;
        m.size += leaves[b->keys[i]]->count * sizeof(t_body);
    }
    for (int i = 0; i < b->cellcount; i++)
    {
        m.size += sizeof(int) * 2;
        m.size += b->cells[i]->count * sizeof(cl_float4);
        m.size += b->matches_counts[i] * sizeof(int);
    }
    m.data = malloc(m.size);
    size_t offset = 0;
    memcpy(m.data + offset, &(b->keycount), sizeof(int));
    offset += sizeof(int);
    
    memcpy(m.data + offset, &(b->cellcount), sizeof(int));
    offset += sizeof(int);
    for (int i = 0; i < b->keycount; i++)
    {
        memcpy(m.data + offset, &(b->keys[i]), sizeof(int));
        offset += sizeof(int);
        memcpy(m.data + offset, &(leaves[b->keys[i]]->count), sizeof(int));
        offset += sizeof(int);
        memcpy(m.data + offset, leaves[b->keys[i]]->bodies, leaves[b->keys[i]]->count * sizeof(t_body));
        offset += leaves[b->keys[i]]->count * sizeof(t_body);
    }
    for (int i = 0; i < b->cellcount; i++)
    {
        memcpy(m.data + offset, &(b->matches_counts[i]), sizeof(int));
        offset += sizeof(int);
        memcpy(m.data + offset, b->matches[i], b->matches_counts[i] * sizeof(int));
        offset += b->matches_counts[i] * sizeof(int);
        memcpy(m.data + offset, &(b->cells[i]->count), sizeof(int));
        offset += sizeof(int);
        for (int j = 0; j < b->cells[i]->count; j++)
        {
            memcpy(m.data + offset, &(b->cells[i]->bodies[j].position), sizeof(cl_float4));
            offset += sizeof(cl_float4);
        }
    }
    return (m);
}

void    bundle_all(t_dispatcher *dispatcher, t_tree **leaves)
{
    int lcount = count_tree_array(leaves);
    int wcount = dispatcher->worker_cnt;
    int leaves_per_bundle = (int)ceil((float)lcount / (float)wcount);
    static int bundle_id = 0;

    dispatcher->cells = leaves;
    dispatcher->total_workunits = lcount;
    dispatcher->cell_count = lcount;
    for (int i = 0; i * leaves_per_bundle < lcount; i++)
    {
        t_bundle *b = bundle_leaves(leaves, i * leaves_per_bundle, leaves_per_bundle);
        b->id = bundle_id++;
        queue_enqueue(&dispatcher->bundles, queue_create_new(b));
        sem_post(dispatcher->start_sending);
    }
}

///////////////////////////////////////////////////////////////
///////////////       Center of Gravity         ///////////////
///////////////////////////////////////////////////////////////

/*
    only leaf cells deternmine their CoG from particles;
    parent cells use their childrens' centers.

    the code is a little more awkward than would normally be
    necessary, due to handling negative masses for the Janus model.
*/

static cl_float4 center_add(cl_float4 total, cl_float4 add)
{
    //method for tallying running total for center of gravity
    add.w = fabs(add.w);
    add.x *= add.w;
    add.y *= add.w;
    add.z *= add.w;
    return (cl_float4){total.x + add.x, total.y + add.y, total.z + add.z, total.w + add.w};
}

static t_body COG_from_children(t_tree **children)
{
    cl_float4 center = (cl_float4){0,0,0,0};
    float real_total = 0;
    for (int i = 0; i < 8; i++)
    {
        center.x += children[i]->as_single->bodies[0].position.x * children[i]->as_single->bodies[0].velocity.w;
        center.y += children[i]->as_single->bodies[0].position.y * children[i]->as_single->bodies[0].velocity.w;
        center.z += children[i]->as_single->bodies[0].position.z * children[i]->as_single->bodies[0].velocity.w;
        center.w += children[i]->as_single->bodies[0].velocity.w;
        real_total += children[i]->as_single->bodies[0].position.w;
    }
    t_body b;
    b.velocity = (cl_float4){0, 0, 0, center.w};
    center.x /= center.w;
    center.y /= center.w;
    center.z /= center.w;
    center.w = real_total;
    b.position = center;
    return b;
}

static t_body COG_from_bodies(t_body *bodies, int count)
{
    cl_float4 center = (cl_float4){0,0,0,0};
    if (count == 0)
        return (t_body){center, center};
    float real_total = 0;
    for (int i = 0; i < count; i++)
    {
        center = center_add(center, bodies[i].position);
        real_total += bodies[i].position.w;
    }
    t_body b;
    b.velocity = (cl_float4){0, 0, 0, center.w};
    center.x /= center.w;
    center.y /= center.w;
    center.z /= center.w;
    center.w = real_total;
    b.position = center;
    return b;
}

/*
    for a cell, create a simplified version of it
    where all particles are replaced with a single
    particle at their center of gravity
*/

static t_tree *make_as_single(t_tree *c)
{
    t_body b;
    if (!c->children)
       b = COG_from_bodies(c->bodies, c->count);
    else
        b = COG_from_children(c->children);
    t_tree *s = calloc(1, sizeof(t_tree));
    s->parent = NULL;
    s->children = NULL;
    s->count = 1;
    s->as_single = NULL;
    s->bodies = calloc(1, sizeof(t_body));
    s->bodies[0] = b;
    return (s);
}

///////////////////////////////////////////////////////////////
///////////////Simple Stuff and Helper Functions///////////////
///////////////////////////////////////////////////////////////

//functions related to the octree structure
t_tree *new_tnode(t_body *bodies, int count, t_tree *parent)
{
    t_tree *node = (t_tree *)calloc(1, sizeof(t_tree));
    node->bodies = bodies;
    node->count = count;
    node->parent = parent;
    node->children = NULL;
    return (node);
}

int node_depth(t_tree *node)
{
    int depth = 0;
    while (node->parent)
    {
        node = node->parent;
        depth++;
    }
    return depth;
}

static void free_tree(t_tree *t)
{
    if (!t)
        return;
    if (t->children)
        for (int i = 0; i < 8; i++)
            free_tree(t->children[i]);
    free(t->neighbors);
    free(t->children);
    if (t->as_single)
    {
        free(t->as_single->bodies);
        free(t->as_single);
    }
    free(t);
}

//create the bounds object for a child cell.
//use 3-bit morton substring to know which octant
//and divide parent bounds appropriately

#define xmid parent.xmax - (parent.xmax - parent.xmin) / 2
#define ymid parent.ymax - (parent.ymax - parent.ymin) / 2
#define zmid parent.zmax - (parent.zmax - parent.zmin) / 2

t_bounds bounds_from_code(t_bounds parent, unsigned int code)
{
    t_bounds bounds;
    if(code >> 2)
    {
        bounds.xmin = xmid;
        bounds.xmax = parent.xmax;
    }
    else
    {
        bounds.xmin = parent.xmin;
        bounds.xmax = xmid;
    }
    code %= 4;
    if (code >> 1)
    {
        bounds.ymin = ymid;
        bounds.ymax = parent.ymax;
    }
    else
    {
        bounds.ymin = parent.ymin;
        bounds.ymax = ymid;
    }
    code %= 2;
    if (code)
    {
        bounds.zmin = zmid;
        bounds.zmax = parent.zmax;
    }
    else
    {
        bounds.zmin = parent.zmin;
        bounds.zmax = zmid;
    }
    return (bounds);
}

static t_bounds bounds_from_bodies(t_body *bodies, int count)
{
    //at the start of making the tree, we need a box that bounds all the bodies.
    float xmin = 0, xmax = 0;
    float ymin = 0, ymax = 0;
    float zmin = 0, zmax = 0;

    for (int i = 0; i < count; i++)
    {
        //expand bounds if needed
        if (bodies[i].position.x < xmin)
            xmin = bodies[i].position.x;
        if (bodies[i].position.x > xmax)
            xmax = bodies[i].position.x;

        if (bodies[i].position.y < ymin)
            ymin = bodies[i].position.y;
        if (bodies[i].position.y > ymax)
            ymax = bodies[i].position.y;

        if (bodies[i].position.z < zmin)
            zmin = bodies[i].position.z;
        if (bodies[i].position.z > zmax)
            zmax = bodies[i].position.z;
    }

    //bounds must be a cube.
    float min = xmin;
    float max = xmax;
    if (ymin < min)
        min = ymin;
    if (zmin < min)
        min = zmin;
    if (ymax > max)
        max = ymax;
    if (zmax > max)
        max = zmax;
    return ((t_bounds){min, max, min, max, min, max});
}

static cl_float4 vadd(cl_float4 a, cl_float4 b)
{
    //add two cl_float4 vectors.
    return ((cl_float4){a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w});
}

static int count_tree_array(t_tree **arr)
{
    int count;
     if (!arr)
        return 0;
    for (count = 0; arr[count]; count++)
        ;
    return (count);
 }

static cl_float4 midpoint_from_bounds(t_bounds b)
{
    return (cl_float4){(b.xmax - b.xmin) / 2, (b.ymax - b.ymin) / 2, (b.zmax - b.zmin) / 2};
}
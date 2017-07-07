/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   do_workunit.c                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: pmclaugh <pmclaugh@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2017/07/06 18:40:33 by pmclaugh          #+#    #+#             */
/*   Updated: 2017/07/07 01:44:40 by pmclaugh         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "worker.h"
#include "err_code.h"

static int count_bodies(t_body **bodies)
{
    int i = 0;
    if (!bodies)
        return (0);
    while (bodies[i])
        i++;
    return (i);
}

void print_cl4(cl_float4 v)
{
    printf("x: %f y: %f z: %f w:%f\n", v.x, v.y, v.z, v.w);
}


static char *load_cl_file(char *filename)
{
    char *source;
    int fd;

    source = (char *)calloc(1,8192);
    fd = open(filename, O_RDONLY);
    if (fd < 0)
    {
        printf("could not find file\n");
        exit(1);
    }
    read(fd, source, 8192);
    return (source);
}

// Adapted from Hands On OpenCL
static t_context *setup_context(void)
{
     cl_uint numPlatforms;
     t_context *c = (t_context *)calloc(1, sizeof(t_context));
     int err;

    // Find number of platforms
    err = clGetPlatformIDs(0, NULL, &numPlatforms);

    // Get all platforms
    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    checkError(err, "Getting platforms");

    // Secure a GPU
    for (int i = 0; i < numPlatforms; i++)
    {
        err = clGetDeviceIDs(Platform[i], DEVICE, 1, &(c->device_id), NULL);
        if (err == CL_SUCCESS)
        {
            break;
        }
    }

    if (c->device_id == NULL)
        checkError(err, "Finding a device");

    // Create a compute context
    c->context = clCreateContext(0, 1, &(c->device_id), NULL, NULL, &err);
    checkError(err, "Creating context");

    // Create a command queue
    c->commands = clCreateCommandQueue(c->context, (c->device_id), 0, &err);
    checkError(err, "Creating command queue");
    printf("setup context\n");
    return (c);
}

static void free_context(t_context *c)
{
    clReleaseCommandQueue(c->commands);
    clReleaseContext(c->context);
}

// Adapted from Hands On OpenCL
static cl_kernel   make_kernel(t_context *c, char *sourcefile, char *name)
{
    cl_kernel k;
    cl_program p;
    int err;
    char *source;

    source = load_cl_file(sourcefile);
    p = clCreateProgramWithSource(c->context, 1, (const char **) & source, NULL, &err);
    checkError(err, "Creating program");

    // Build the program
    err = clBuildProgram(p, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(p, c->device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(0);
    }

    // Create the compute kernel from the program
    k = clCreateKernel(p, name, &err);
    checkError(err, "Creating kernel");
    clReleaseProgram(p);
    printf("made kernel\n");
    //free(source);
    return (k);
}

/*
    This is the core computation. The code for the GPU kernel is in nxm2.cl.
    OpenCL is very careful and verbose, so the function is quite long, but essentially
    it is just allocating space on the GPU, transferring the relevant data,
    starting the kernel, waiting for it to finish, and pulling the data off.

    Calling clFinish() for each workunit is unfortunate, but this was primarily developed
    on OS X, which does not allow CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE. We tested
    not calling to finish (hence the cl_events everywhere), and instead returning a cl_event 
    that the sender thread would wait on, but in practice it offered only minimal speedup 
    and occasionally caused instability.
*/

static t_body *crunch_NxM(cl_float4 *N, cl_float4 *V, cl_float4 *M, size_t ncount, size_t mcount)
{
    static t_context   *context;
    static cl_kernel   k_nbody;
    int err;

    if (context == NULL)
        context = setup_context();
    if (k_nbody == NULL)
        k_nbody = make_kernel(context, "nxm2.cl", "nbody");

    cl_float4 *output_p = (cl_float4 *)calloc(ncount, sizeof(cl_float4));
    cl_float4 *output_v = (cl_float4 *)calloc(ncount, sizeof(cl_float4));

    //printf("NxM: %lu x %lu\n", ncount, mcount);
    //device-side data
    cl_mem      d_N_start;
    cl_mem      d_M;
    cl_mem      d_V_start;
    cl_mem      d_V_end;
    cl_mem      d_N_end;
    //inputs
    d_N_start = clCreateBuffer(context->context, CL_MEM_READ_ONLY, sizeof(cl_float4) * ncount, NULL, NULL);
    d_M = clCreateBuffer(context->context, CL_MEM_READ_ONLY, sizeof(cl_float4) * mcount, NULL, NULL);
    d_V_start = clCreateBuffer(context->context, CL_MEM_READ_ONLY, sizeof(cl_float4) * ncount, NULL, NULL);
    //outputs
    d_V_end = clCreateBuffer(context->context, CL_MEM_WRITE_ONLY, sizeof(cl_float4) * ncount, NULL, NULL);
    d_N_end = clCreateBuffer(context->context, CL_MEM_WRITE_ONLY, sizeof(cl_float4) * ncount, NULL, NULL);

    //all of that happens instantly and therefore we don't need events involved

    //copy over initial data to device locations
    cl_event eN, eM, eV;
    clEnqueueWriteBuffer(context->commands, d_N_start, CL_FALSE, 0, sizeof(cl_float4) * ncount, N, 0, NULL, &eN);
    clEnqueueWriteBuffer(context->commands, d_M, CL_FALS, 0, sizeof(cl_float4) * mcount, M, 0, NULL, &eM);
    clEnqueueWriteBuffer(context->commands, d_V_start, CL_FALSE, 0, sizeof(cl_float4) * ncount, V, 0, NULL, &eV);

    cl_event loadevents[3] = {eN, eM, eV};

    //set some parameters up
    size_t tps = 32;
    size_t global = ncount * tps;
    size_t mscale = mcount;
    size_t local = GROUPSIZE < global ? GROUPSIZE : global;
    float soften = SOFTENING;
    float timestep = TIME_STEP;
    float grav = G;
    size_t count = ncount;

    //connect kernel arguments to the kernel
    clSetKernelArg(k_nbody, 0, sizeof(cl_mem), &d_N_start);
    clSetKernelArg(k_nbody, 1, sizeof(cl_mem), &d_N_end);
    clSetKernelArg(k_nbody, 2, sizeof(cl_mem), &d_M);
    clSetKernelArg(k_nbody, 3, sizeof(cl_mem), &d_V_start);
    clSetKernelArg(k_nbody, 4, sizeof(cl_mem), &d_V_end);
    clSetKernelArg(k_nbody, 5, sizeof(cl_float4) * GROUPSIZE, NULL);
    clSetKernelArg(k_nbody, 6, sizeof(float), &soften);
    clSetKernelArg(k_nbody, 7, sizeof(float), &timestep);
    clSetKernelArg(k_nbody, 8, sizeof(float), &grav);
    clSetKernelArg(k_nbody, 9, sizeof(unsigned int), &global);
    clSetKernelArg(k_nbody, 10, sizeof(unsigned int), &mscale);
    clSetKernelArg(k_nbody, 11, sizeof(unsigned int), &tps);

    cl_event compute;
    cl_event offN, offV;

    err = clEnqueueNDRangeKernel(context->commands, k_nbody, 1, NULL, &global, &local, 3, loadevents, &compute);
    clEnqueueReadBuffer(context->commands, d_N_end, CL_FALSE, 0, sizeof(cl_float4) * count, output_p, 1, &compute, &offN);
    clEnqueueReadBuffer(context->commands, d_V_end, CL_FALSE, 0, sizeof(cl_float4) * count, output_v, 1, &compute, &offV);
    clFinish(context->commands);

    clReleaseMemObject(d_N_start);
    clReleaseMemObject(d_N_end);
    clReleaseMemObject(d_M);
    clReleaseMemObject(d_V_start);
    clReleaseMemObject(d_V_end);

    clReleaseEvent(eN);
    clReleaseEvent(eM);
    clReleaseEvent(eV);
    clReleaseEvent(compute);
    clReleaseEvent(offN);
    clReleaseEvent(offV);

    t_body *ret = (t_body *)malloc(sizeof(t_body) * ncount);
    for (int i = 0; i < ncount; i++)
    {
        ret[i].position = output_p[i];
        ret[i].velocity = output_v[i];
    }
    if (output_p) free(output_p);
    if (output_v) free(output_v);
    return (ret);
}

void do_workunit(t_workunit *w)
{
    //this is just a convenient wrapper function for the above function.
    w->local_bodies = crunch_NxM(w->N, w->V, w->M, w->localcount + w->npadding, w->neighborcount + w->mpadding);
    w->neighborcount = 0;
    if (w->N) free(w->N);
    if (w->M) free(w->M);
    if (w->V) free(w->V);
}

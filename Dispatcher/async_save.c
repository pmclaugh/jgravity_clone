/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   async_save.c                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: pmclaugh <pmclaugh@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2017/05/30 18:34:25 by pmclaugh          #+#    #+#             */
/*   Updated: 2017/07/07 01:30:13 by pmclaugh         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "dispatcher.h"

/*
	We can write the output file as we receive workunits back from the workers.
	Originally we were waiting til everything had come back, then writing.
	This was fine for small problem sizes but for 2^28 particles the output file is many GB
*/

void setup_async_file(t_dispatcher *dispatcher)
{
	char *filename;
	asprintf(&filename, "%s-%d.jgrav", dispatcher->name, dispatcher->ticks_done);
	pthread_mutex_lock(&dispatcher->output_mutex);
	dispatcher->fp = fopen(filename, "w");
	fwrite(&dispatcher->dataset->particle_cnt, sizeof(long), 1, dispatcher->fp);
	fseek(dispatcher->fp, sizeof(cl_float4) * dispatcher->dataset->particle_cnt, SEEK_SET);
	fputc('\0', dispatcher->fp);
	fseek(dispatcher->fp, 0, SEEK_SET);
	pthread_mutex_unlock(&dispatcher->output_mutex);
	free(filename);
}

void async_save(t_dispatcher *dispatcher, unsigned long offset, t_WU *wu)
{
	pthread_mutex_lock(&dispatcher->output_mutex);
	fseek(dispatcher->fp, offset * sizeof(cl_float4) + sizeof(long), SEEK_SET);
	for (int i = 0; i < wu->localcount; i++)
		fwrite(&wu->local_bodies[i].position, sizeof(cl_float4), 1, dispatcher->fp);
	pthread_mutex_unlock(&dispatcher->output_mutex);
}
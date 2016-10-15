/*
 *	Convert CAD format of OFF in PASCAL3D+ dataset into binvox for 3D-R2N2
 *	Copyright (C) 2016  Xingyou Chen <niatlantice@gmail.com>
 *
 *	This program is free software; you can redistribute it and/or modify
 *	it under the terms of the GNU General Public License as published by
 *	the Free Software Foundation; either version 2 of the License, or
 *	(at your option) any later version.
 *
 *	This program is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *	GNU General Public License for more details.
 *
 *	You should have received a copy of the GNU General Public License
 *	along with this program; if not, write to the Free Software
 *	Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */
/* BUILD: ${CC} -o off2binvox off2binvox.c -lm */
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <math.h>

struct vertex_s {
	float x;
	float y;
	float z;
} *vertex = NULL;

struct mesh_s {
	int x;
	int y;
	int z;
} *mesh = NULL;

struct voxel_s {
	short dim;
	struct vertex_s trans;
	float scale;
	unsigned verts;
	unsigned meshs;
	unsigned char *data;
} voxel;

// Initialize voxel
void init_voxel(void)
{
	voxel.dim = 32;
	voxel.trans.x = voxel.trans.y = voxel.trans.z = 0;
	voxel.scale = 1;
	voxel.data = (unsigned char *)malloc(voxel.dim * voxel.dim * voxel.dim);
	if (voxel.data == NULL)
		exit(2);
	memset(voxel.data, 0, voxel.dim * voxel.dim * voxel.dim);
}

int read_off(char *off_file)
{
	FILE *off_fd = stdin;
	int vet_cnt = 0, mesh_cnt = 0;
	int idx, angles;		// For PASCAL3D, meshes are all triangular
	unsigned char line_buf[256];	// Line column should never exceed this threshold

	if (off_file != NULL) {
		off_fd = fopen(off_file, "r");
		if (off_fd == NULL)
			return 3;
	}
	fscanf(off_fd, "%s", line_buf);
	if (strncmp(line_buf, "OFF", 4)) {
		printf("[ERROR] Input is not in OFF format.\n");
		return 3;
	}
	fscanf(off_fd, "%d %d %d\n", &vet_cnt, &mesh_cnt, &angles);
	vertex = (struct vertex_s *)malloc(sizeof(struct vertex_s) * vet_cnt);
	mesh = (struct mesh_s *)malloc(sizeof(struct mesh_s) * mesh_cnt);
	voxel.verts = vet_cnt; voxel.meshs = mesh_cnt;

	for (idx = 0; idx < vet_cnt; idx++)
		fscanf(off_fd, "%f %f %f\n", &((vertex + idx)->x), &((vertex + idx)->y), &((vertex + idx)->z));
	for (idx = 0; idx < mesh_cnt; idx++) {
		fscanf(off_fd, "%d %d %d %d\n", &angles, &((mesh + idx)->x), &((mesh + idx)->y), &((mesh + idx)->z));
		// Important: vertex index need be advanced by 1, seemed that off format starts from 0
		(mesh + idx)->x = (mesh + idx)->x + 1;
		(mesh + idx)->y = (mesh + idx)->y + 1;
		(mesh + idx)->z = (mesh + idx)->z + 1;
	}

	fclose(off_fd);
	return EXIT_SUCCESS;
}

// Fit entire model in a (dim x dim x dim) box
void coord_trans(void)
{
	float min_x = FLT_MAX, max_x = FLT_MIN;
	float min_y = FLT_MAX, max_y = FLT_MIN;
	float min_z = FLT_MAX, max_z = FLT_MIN;
	float range, range_x, range_y, range_z, zero_x, zero_y, zero_z, scale;

	// Get cubic box of model
	for (int i = 0; i < voxel.verts; i++) {
		if (vertex[i].x < min_x)
			min_x = vertex[i].x;
		else if (vertex[i].x > max_x)
			max_x = vertex[i].x;
		if (vertex[i].y < min_y)
			min_y = vertex[i].y;
		else if (vertex[i].y > max_y)
			max_y = vertex[i].y;
		if (vertex[i].z < min_z)
			min_z = vertex[i].z;
		else if (vertex[i].z > max_z)
			max_z = vertex[i].z;
	}
	range_x = max_x - min_x;
	range_y = max_y - min_y;
	range_z = max_z - min_z;
	range = (range_x > range_y) ? ((range_x > range_z) ? range_x : range_z) : ((range_y > range_z) ? range_y : range_z);
	zero_x = min_x - (range - range_x) / 2;
	zero_y = min_y - (range - range_y) / 2;
	zero_z = min_z - (range - range_z) / 2;
	scale = voxel.dim / range;

	for (int i = 0; i < voxel.verts; i++) {
		vertex[i].x = (vertex[i].x - zero_x) * scale;
		vertex[i].y = (vertex[i].y - zero_y) * scale;
		vertex[i].z = (vertex[i].z - zero_z) * scale;
	}
}

void calc_voxel(void)
{
	struct mesh_s cube_s = {INT_MAX, INT_MAX, INT_MAX}, cube_e = {0, 0, 0};
	struct vertex_s *v[3];

	for (int idx = 0; idx < voxel.meshs; idx++) {
		// Calculate bounding cube of a mesh and align bounding cube to grid
		v[0] = &vertex[mesh[idx].x];
		v[1] = &vertex[mesh[idx].y];
		v[2] = &vertex[mesh[idx].z];

		for (int j = 0; j < 3; j++) {
			if (v[j]->x < cube_s.x)
				cube_s.x = floor(v[j]->x);
			else if (v[j]->x > cube_e.x)
				cube_e.x = ceil(v[j]->x);

			if (v[j]->y < cube_s.y)
				cube_s.y = floor(v[j]->y);
			else if (v[j]->y > cube_e.y)
				cube_e.y = ceil(v[j]->y);

			if (v[j]->z < cube_s.z)
				cube_s.z = floor(v[j]->z);
			else if (v[j]->z > cube_e.z)
				cube_e.z = ceil(v[j]->z);
		}

		// Translate bounding cube into voxel segment
		for (int i = cube_s.x; i < cube_e.x; i++)
			for (int j = cube_s.y; j < cube_e.y; j++)
				for (int k = cube_s.z; k < cube_e.z; k++)
					voxel.data[i * voxel.dim * voxel.dim + k * voxel.dim + j] = 1;
	}
}

void save_binvox(FILE *vox_fd)
{
	unsigned char cur_stat, cur_cnt;
	fprintf(vox_fd, "#binvox 1\ndim %d %d %d\ntranslate %f %f %f\nscale %f\ndata\n", voxel.dim, voxel.dim, voxel.dim,
		voxel.trans.x, voxel.trans.y, voxel.trans.z, voxel.scale);

	cur_stat = voxel.data[0];
	cur_cnt = 0;
	for (int idx = 0; idx < (voxel.dim*voxel.dim*voxel.dim); idx++) {
		if (cur_stat == voxel.data[idx]) {
			cur_cnt++;
			if (cur_cnt == 255) {
				fputc(cur_stat, vox_fd);
				fputc(cur_cnt, vox_fd);
				cur_cnt = 0;
			}
		} else {
			if (cur_cnt > 0) {
				fputc(cur_stat, vox_fd);
				fputc(cur_cnt, vox_fd);
			}
			cur_stat = voxel.data[idx];
			cur_cnt = 1;
		}
	}
}

int main(int argc, char *argv[])
{
	int ret_val;
	char *off_file = NULL;
	FILE *vox_fd = stdout;

	if (argc == 1)
		return 4;
	if (argc > 1)
		off_file = argv[1];
	if (argc > 2)
		vox_fd = fopen(argv[2], "w");
	if (vox_fd == NULL)
		return 2;	// Ramdomly chosen

	ret_val = read_off(off_file);
	if (ret_val != EXIT_SUCCESS)
		return ret_val;

	init_voxel();
	coord_trans();
	calc_voxel();
	save_binvox(vox_fd);

	fclose(vox_fd);
	free(voxel.data);
	free(mesh);
	free(vertex);
	return EXIT_SUCCESS;
}

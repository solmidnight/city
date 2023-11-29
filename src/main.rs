use bevy::prelude::*;
use bevy::render::color::Color;
use bevy::render::mesh::Indices;
use bevy::render::mesh::VertexAttributeValues;
use bevy::render::render_resource::PrimitiveTopology;
use bitflags::bitflags;

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Block {
    Void,
    Air,
    Stone,
}

pub trait Structure {
    fn new() -> Self
    where
        Self: Sized;

    fn get(&self, position: UVec3) -> Block;
    fn set(&mut self, position: UVec3, block: Block);

    fn size(&self) -> UVec3;

    fn linearize(&self, position: UVec3) -> usize {
        let UVec3 { y: sy, x: sx, .. } = self.size();
        let UVec3 { x, y, z } = position;
        ((z * sy + y) * sx + x) as usize
    }

    fn delinearize(&self, index: usize) -> UVec3 {
        let UVec3 {
            x: sx,
            y: sy,
            z: sz,
        } = self.size();
        let mut idx = index as u32;
        let z = idx / (sx * sy);
        idx -= (z * sx * sy);
        let y = idx / sx;
        let x = idx % sx;
        UVec3 { x, y, z }
    }

    fn count(&self) -> usize {
        let UVec3 { x, y, z } = self.size();
        (x * y * z) as usize
    }
}

pub struct Chunk {
    blocks: Vec<Block>,
}

impl Structure for Chunk {
    fn new() -> Self
    where
        Self: Sized,
    {
        Chunk {
            blocks: vec![Block::Void; 64 * 64 * 64],
        }
    }

    fn get(&self, position: UVec3) -> Block {
        let index = self.linearize(position);
        self.blocks[index]
    }

    fn set(&mut self, position: UVec3, block: Block) {
        let index = self.linearize(position);
        self.blocks[index] = block
    }

    fn size(&self) -> UVec3 {
        UVec3::new(64, 64, 64)
    }
}

bitflags! {
    #[derive(PartialEq, Eq, Clone, Copy)]
    struct Direction: usize {
        const LEFT =    0b00000001;
        const RIGHT =   0b00000010;
        const DOWN =    0b00000100;
        const UP =      0b00001000;
        const BACK =    0b00010000;
        const FOWARD =  0b00100000;
        const ALL =  0b00111111;
    }
}

fn cube_mesh_parts(
    position: Vec3,
    directions: Direction,
    vertices: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let cube_vertices = [
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        [
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
    ];

    let cube_normals = [
        [
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        [
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
        ],
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
    ];

    let cube_indices = [
        [4, 5, 7, 5, 6, 7],
        [0, 3, 1, 1, 3, 2],
        [12, 13, 15, 13, 14, 15],
        [8, 11, 9, 9, 11, 10],
        [20, 21, 23, 21, 22, 23],
        [16, 19, 17, 17, 19, 18],
    ];

    for current_direction in (0..6)
        .map(|x| 1 << x)
        .map(Direction::from_bits)
        .map(Option::unwrap)
    {
        if current_direction & directions == Direction::empty() {
            continue;
        }
        let index = current_direction.bits().trailing_zeros() as usize;

        let count = vertices.len();

        vertices.extend(
            cube_vertices[index]
                .iter()
                .map(|unit| (Vec3::from_array(*unit) + position).to_array()),
        );
        normals.extend(cube_normals[index].iter());
        indices.extend(cube_indices[index].iter().map(|i| (count + i % 4) as u32))
    }
}

#[rustfmt::skip]
fn create_structure_mesh(structure: &dyn Structure) -> Mesh {
    let mut vertices = vec![];
    let mut normals = vec![];
    let mut indices = vec![];

    let UVec3 { x: sx, y: sy, z: sz } = structure.size();

    for index in 0..structure.count() {
        let position = structure.delinearize(index);
        if !matches!(structure.get(position), Block::Air) {
            let mut dir_iter = (0..6).map(|x| 1 << x).map(Direction::from_bits).map(Option::unwrap);
            let mut directions = Direction::empty();
            for d in 0..3 {
                for n in (-1..=1).step_by(2) {
                    let current_direction = dir_iter.next().unwrap();
                    let mut normal = IVec3::default();
                    normal[d] = n;
                    let normal = normal.as_uvec3();
                    let neighbor = position + normal;
                    if neighbor.x >= sx || neighbor.y >= sy || neighbor.z >= sz {
                            directions |= current_direction;
                    }
                    if neighbor.x < sx && neighbor.y < sy && neighbor.z < sz {
                        if matches!(structure.get(neighbor), Block::Air) {
                            directions |= current_direction;
                        }
                    }
                }
            } 
            cube_mesh_parts(position.as_vec3(), directions, &mut vertices, &mut normals, &mut indices);
        }
    }
dbg!(vertices.len());
    Mesh::new(PrimitiveTopology::TriangleList)
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
            vertices
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        normals
    )
    .with_indices(Some(Indices::U32(indices)))
}

fn camera(mut query: Query<(&Camera, &mut Transform)>, keys: Res<Input<KeyCode>>, time: Res<Time>) {
    let (camera, mut transform) = query.single_mut();
    let speed = 10.4;
    let lateral_direction = IVec3 { 
        x: keys.pressed(KeyCode::D) as i32 - keys.pressed(KeyCode::A) as i32,
        y: 0,
        z: keys.pressed(KeyCode::S) as i32 - keys.pressed(KeyCode::W) as i32,
    };
    let vertical_direction =  IVec3 { 
        x: 0,
        y: keys.pressed(KeyCode::Space) as i32 - keys.pressed(KeyCode::ShiftLeft) as i32,
        z: 0,
    };
    let rotation = Quat::from_axis_angle(Vec3::Y, transform.rotation.to_euler(EulerRot::XYZ).1);
    let movement = speed * time.delta_seconds() * (rotation * lateral_direction.as_vec3() + vertical_direction.as_vec3()).normalize_or_zero();
    transform.translation += movement;
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    dbg!("yo");
    let camera_transform =
        Transform::from_xyz(32.0, 40.0, 32.0);
    commands.spawn(Camera3dBundle {
        transform: camera_transform,
        ..default()
    });
    let mut light_transform = Transform::from_xyz(1000.0, 10000.0, 1000.0);
    light_transform.look_at(Vec3::ZERO, Vec3::Y);
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight { color: Color::Rgba { red: 1.0, green: 0.996, blue: 0.976, alpha: 1.0 }, illuminance: 10000.0, shadows_enabled: true, ..default() },
        transform: light_transform,
        ..default()
    });

    let mut chunk = Chunk {
        blocks: vec![Block::Void; 64 * 64 * 64],
    };
    let perlin = noise::Fbm::<noise::Perlin>::new(400);
    for z in 0..64 {
        for x in 0..64 {
            for y in 0..64 {
                use noise::NoiseFn;
                let density = perlin.get([x as f64 * 0.02, y as f64 * 0.02, z as f64 * 0.02]);
                let density_mod = (32isize - y as isize) as f64 * 0.035;
                chunk.set(
                    UVec3 { x, y, z },
                    if density + density_mod > 0.0 {
                        Block::Stone
                    } else {
                        Block::Air
                    },
                );
            }
        }
    }

    let cube_mesh_handle: Handle<Mesh> = meshes.add(create_structure_mesh(&chunk));

    commands.spawn(
        (PbrBundle {
            mesh: cube_mesh_handle,
            material: materials.add(StandardMaterial {
                base_color: Color::Rgba {
                    red: 1.0,
                    green: 1.0,
                    blue: 1.0,
                    alpha: 1.0,
                },
                ..default()
            }),
            ..default()
        }),
    );
}

fn main() {
    dbg!("yo");
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, camera)
        .run();
}
